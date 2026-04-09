#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v2
# Hypothesis: Breakout above/below Camarilla pivot levels (H3/L3) on 12h chart with 1d trend filter (EMA 50).
# Only take long when price > 1d EMA(50), short when price < 1d EMA(50).
# Exit when price crosses 1d EMA(50) in opposite direction.
# Uses volume confirmation (volume > 1.5x 20-period average) and volatility filter (ATR < 3% of price).
# Target: 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to trend filter + volume/volatility filters reducing whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v2"
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
    
    # Camarilla pivot levels (based on previous day's range)
    # For 12h timeframe, we use daily pivot levels
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous day's high, low, close
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Load 1d data ONCE before loop for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros_like(close_1d, dtype=float)
    ema_1d[0] = close_1d[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
        if np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.03 * close[i]  # ATR less than 3% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: price > 1d EMA for longs, price < 1d EMA for shorts
        trend_long = close[i] > ema_1d_aligned[i]
        trend_short = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Camarilla H3 with volume confirmation, volatility filter, and trend filter
            if close[i] > camarilla_h3[i] and vol_ok and vol_filter and trend_long:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Camarilla L3 with volume confirmation, volatility filter, and trend filter
            elif close[i] < camarilla_l3[i] and vol_ok and vol_filter and trend_short:
                position = -1
                signals[i] = -0.25
    
    return signals