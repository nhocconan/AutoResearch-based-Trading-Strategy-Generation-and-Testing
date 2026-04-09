#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with volume confirmation and ATR stoploss
# - Uses 1d HTF for prior day's high/low to calculate Williams %R on 12h data
# - Long when Williams %R < -80 (oversold) with volume > 1.5x 20-period average
# - Short when Williams %R > -20 (overbought) with volume > 1.5x 20-period average
# - ATR(14) trailing stop: exit long at 2.0x ATR below lowest low since entry
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)
# - Williams %R captures extreme reversals, volume filter reduces false signals
# - Works in both bull and bear markets by fading exhaustion moves

name = "12h_1d_williamsr_meanrev_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    lowest_low_since_entry = 0.0
    highest_high_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i]) or vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from lowest low
            if close[i] < lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                lowest_low_since_entry = 0.0
                highest_high_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from highest high
            if close[i] > highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                lowest_low_since_entry = 0.0
                highest_high_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extremes + volume confirmation
            if volume_confirmed:
                # Long entry: Williams %R < -80 (oversold)
                if williams_r_aligned[i] < -80:
                    position = 1
                    lowest_low_since_entry = low[i]
                    highest_high_since_entry = high[i]
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought)
                elif williams_r_aligned[i] > -20:
                    position = -1
                    lowest_low_since_entry = low[i]
                    highest_high_since_entry = high[i]
                    signals[i] = -0.25
    
    return signals