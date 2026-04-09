#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR trailing stop using 12h HTF
# - Uses 12h HTF for prior period's high/low to calculate Donchian(10) channels (shorter for more signals)
# - Long when price closes above upper Donchian with volume > 1.3x 20-period average
# - Short when price closes below lower Donchian with volume > 1.3x 20-period average
# - ATR(10) trailing stop: exit long at 2.0x ATR below highest high since entry
# - Fixed position size 0.25 to control drawdown
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)
# - Volume filter reduces false breakouts, ATR stop manages risk
# - Works in both bull and bear markets by capturing breakouts in trending phases

name = "4h_12h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 10-period Donchian channels on 12h data
    # Upper band: highest high of last 10 periods
    # Lower band: lowest low of last 10 periods
    upper_10 = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    lower_10 = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 4h timeframe (wait for completed 12h bar)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_10)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_10)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR (10-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation
            if volume_confirmed:
                # Long entry: price closes above upper Donchian
                if close[i] > upper_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price closes below lower Donchian
                elif close[i] < lower_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals