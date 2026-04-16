# 1d_WeeklyPivot_Bias_DailyBreakout_Volume
# Hypothesis: Weekly pivot points (PP, R1, S1) provide strong support/resistance levels.
# In bull markets, price tends to respect S1 as support and break above PP/R1.
# In bear markets, price tends to respect R1 as resistance and break below PP/S1.
# Combining with daily breakouts and volume filters reduces false signals.
# Timeframe: 1d (daily bars) for lower frequency and less fee drag.
# Uses 1w pivot levels as HTF bias and 1d price/volume for entry timing.
# Target: 30-100 trades over 4 years (7-25/year) to stay within fee limits.
# Works in both bull and bear via pivot bias (long when above weekly PP, short when below).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF for pivot bias) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivot levels to daily timeframe (with proper delay for weekly close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === Daily indicators for entry timing ===
    # Daily ATR for volatility filter
    high_1d = high
    low_1d = low
    close_1d = close
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pp = pp_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        atr_val = atr_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly S1 (support broken)
            if price < s1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly R1 (resistance broken)
            if price > r1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above weekly PP (bullish bias) AND breaks above weekly R1 
            # AND volume spike AND volatility not extreme
            if (price > pp) and (price > r1) and (vol_ratio_val > 2.0) and (atr_val < np.percentile(atr_1d[:i+1], 80)):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below weekly PP (bearish bias) AND breaks below weekly S1 
            # AND volume spike AND volatility not extreme
            elif (price < pp) and (price < s1) and (vol_ratio_val > 2.0) and (atr_val < np.percentile(atr_1d[:i+1], 80)):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_Bias_DailyBreakout_Volume"
timeframe = "1d"
leverage = 1.0