#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Regime
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 1d chart.
Price breaking above R3 with volume spike (>1.5x avg) and in choppy regime (CHOP>61.8) 
indicates breakout from range, good for long. Price breaking below S3 with volume spike
and choppy regime indicates breakdown, good for short. Uses 1w trend filter (price > EMA20 
on weekly for long bias, < EMA20 for short bias) to avoid counter-trend trades. 
Exit on opposite pivot touch (R3/S3) or trend reversal. Discreet sizing 0.25 to minimize 
fee churn. Targets 50-100 trades over 4 years (12-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and chop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), 
    # R2 = C + ((H-L) * 1.1666/2), R1 = C + ((H-L) * 1.0833/2)
    # PP = (H+L+C)/3, S1 = C - ((H-L) * 1.0833/2), etc.
    # We only need R3 and S3 for breakout
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 (using previous day's values)
    # Shift by 1 to use previous day's H,L,C
    h_prev = np.roll(h_1d, 1)
    l_prev = np.roll(l_1d, 1)
    c_prev = np.roll(c_1d, 1)
    # First value will be invalid (rolled from last), handle with min_periods later
    diff = h_prev - l_prev
    r3 = c_prev + (diff * 1.25 / 2)
    s3 = c_prev - (diff * 1.25 / 2)
    
    # Get 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d (CHOP > 61.8 = ranging/choppy)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HHV(high,14) - LLV(low,14))))
    # Simplified: use rolling ATR and range
    tr1 = np.abs(h_1d[1:] - l_1d[:-1])
    tr2 = np.abs(h_1d[1:] - c_1d[:-1])
    tr3 = np.abs(l_1d[1:] - c_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(h_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(l_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log10(14) / (hh - ll + 1e-10))
    
    # Volume confirmation: current volume > 1.5 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all 1d indicators to 1d timeframe (same timeframe, so just shift for completion)
    # For 1d data on 1d timeframe, we need to wait for bar close, so shift by 1
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 1w EMA20 to 1d timeframe (wait for weekly close)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = max(30, 14, 1)  # volume avg(30), chop(14), plus 1 for alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        ema_1w_val = ema_20_1w_aligned[i]
        
        if position == 0:
            # Determine trend bias from 1w EMA20: price > EMA = bullish bias, < EMA = bearish bias
            is_bullish_bias = close_val > ema_1w_val
            is_bearish_bias = close_val < ema_1w_val
            
            # Only trade in choppy/ranging regime (CHOP > 61.8)
            in_choppy_regime = chop_val > 61.8
            
            if is_bullish_bias and in_choppy_regime:
                # Bullish bias: long when price breaks above R3 with volume confirmation
                if (close_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_bearish_bias and in_choppy_regime:
                # Bearish bias: short when price breaks below S3 with volume confirmation
                if (close_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S3 (support) or trend turns bearish
            exit_condition = (close_val < s3_val) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend turns bullish
            exit_condition = (close_val > r3_val) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Regime"
timeframe = "1d"
leverage = 1.0