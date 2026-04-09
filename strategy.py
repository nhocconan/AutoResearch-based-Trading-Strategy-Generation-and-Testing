#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v2
# Hypothesis: Breakout above/below weekly Camarilla pivot levels with daily trend filter (EMA 50).
# Only take long when price > daily EMA(50), short when price < daily EMA(50).
# Exit when price crosses daily EMA(50) in opposite direction.
# Uses volatility filter (ATR < 3% of price) and volume confirmation (volume > 1.5x 20-period avg).
# Target: 7-25 trades/year (30-100 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to trend filter + volatility/volume filters reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Load weekly data ONCE before loop for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_h4 = np.zeros(len(close_1w))
    camarilla_l4 = np.zeros(len(close_1w))
    camarilla_h3 = np.zeros(len(close_1w))
    camarilla_l3 = np.zeros(len(close_1w))
    camarilla_h2 = np.zeros(len(close_1w))
    camarilla_l2 = np.zeros(len(close_1w))
    camarilla_h1 = np.zeros(len(close_1w))
    camarilla_l1 = np.zeros(len(close_1w))
    
    for i in range(len(close_1w)):
        if i == 0:
            camarilla_h4[i] = camarilla_l4[i] = camarilla_h3[i] = camarilla_l3[i] = camarilla_h2[i] = camarilla_l2[i] = camarilla_h1[i] = camarilla_l1[i] = np.nan
        else:
            # Calculate pivot point and range
            pivot = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3
            range_ = high_1w[i-1] - low_1w[i-1]
            
            camarilla_h4[i] = pivot + range_ * 1.1 / 2
            camarilla_l4[i] = pivot - range_ * 1.1 / 2
            camarilla_h3[i] = pivot + range_ * 1.1 / 4
            camarilla_l3[i] = pivot - range_ * 1.1 / 4
            camarilla_h2[i] = pivot + range_ * 1.1 / 6
            camarilla_l2[i] = pivot - range_ * 1.1 / 6
            camarilla_h1[i] = pivot + range_ * 1.1 / 12
            camarilla_l1[i] = pivot - range_ * 1.1 / 12
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    
    # Load daily data ONCE before loop for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align daily EMA to daily timeframe (same timeframe, no alignment needed)
    ema_1d_aligned = ema_1d  # Already on daily timeframe
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.03 * close[i]  # ATR less than 3% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > daily EMA for longs, price < daily EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below daily EMA
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above daily EMA
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Camarilla H4 with volume confirmation, volatility filter, and trend filter
            if close[i] > camarilla_h4_aligned[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Camarilla L4 with volume confirmation, volatility filter, and trend filter
            elif close[i] < camarilla_l4_aligned[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals