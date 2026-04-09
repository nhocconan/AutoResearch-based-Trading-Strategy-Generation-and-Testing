#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with volume spike and chop regime filter
# - Uses 1d HTF to calculate Williams %R (14) from prior completed day
# - Long when Williams %R crosses above -80 from below (oversold bounce) with volume > 2.0x 20-period average
# - Short when Williams %R crosses below -20 from above (overbought rejection) with volume > 2.0x 20-period average
# - Choppiness Index (14) regime filter: only trade when CHOP < 61.8 (trending market)
# - ATR(14) trailing stop: exit at 2.0x ATR from extremes since entry
# - Fixed position size 0.25 to control drawdown
# - Williams %R captures momentum exhaustion, volume confirms institutional participation
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)

name = "4h_1d_williamsr_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior day's Williams %R (14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 4h timeframe (wait for completed 1d bar)
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
    
    # Pre-compute Choppiness Index (14) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high, period) - min(low, period))) / log10(period)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop = np.where(chop_denom > 0, 100 * np.log10(atr_14 / chop_denom) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    williams_r_prev = williams_r_aligned[0] if not np.isnan(williams_r_aligned[0]) else -50
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0 or atr[i] <= 0):
            signals[i] = 0.0
            williams_r_prev = williams_r_aligned[i] if not np.isnan(williams_r_aligned[i]) else williams_r_prev
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        regime_filter = chop[i] < 61.8
        
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
            # Entry logic: Williams %R reversal + volume confirmation + regime filter
            if volume_confirmed and regime_filter:
                williams_r_curr = williams_r_aligned[i]
                williams_r_prev_val = williams_r_prev
                
                # Long entry: Williams %R crosses above -80 from below (oversold bounce)
                if (williams_r_prev_val <= -80 and williams_r_curr > -80 and
                    close[i] > close[i-1]):  # additional price confirmation
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: Williams %R crosses below -20 from above (overbought rejection)
                elif (williams_r_prev_val >= -20 and williams_r_curr < -20 and
                      close[i] < close[i-1]):  # additional price confirmation
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
        
        # Update previous Williams %R for next iteration
        williams_r_prev = williams_r_aligned[i] if not np.isnan(williams_r_aligned[i]) else williams_r_prev
    
    return signals