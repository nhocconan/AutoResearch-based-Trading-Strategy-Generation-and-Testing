# 1d Weekly Pivot Breakout with Volume Confirmation
# Hypothesis: Weekly pivots identify key support/resistance levels. Breakouts above R1 or below S1 with volume confirmation indicate institutional interest and momentum. This strategy works in both bull and bear markets by capturing breakout moves from significant weekly levels, using daily timeframe for lower frequency and reduced fee impact.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while maintaining statistical significance.
# Uses 1-week data for pivot calculation (HTF) and 1d for execution, avoiding look-ahead bias through proper alignment.

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
    
    # === Weekly data (HTF for pivot calculation) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    # Pivot = (H + L + C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    # R2 = P + (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # S2 = P - (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly levels to daily timeframe (wait for weekly bar close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
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
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC (already in datetime64 format)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        atr_val = atr_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly pivot OR RSI-like condition (price < pivot)
            if price < pivot:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly pivot OR price > pivot
            if price > pivot:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price breaks above R1 with volume confirmation AND not excessive volatility
                if (price > r1) and (vol_ratio_val > 1.8) and (atr_val < np.percentile(atr_1d[max(0, i-20):i+1], 80)):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below S1 with volume confirmation AND not excessive volatility
                elif (price < s1) and (vol_ratio_val > 1.8) and (atr_val < np.percentile(atr_1d[max(0, i-20):i+1], 80)):
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

name = "1d_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0