#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# Uses 1h Camarilla pivot levels (H3/L3) from previous hour for breakout signals
# Trend filter: 4h EMA(21) - price must be above EMA for longs, below for shorts
# Volume confirmation: 1h volume > 1.5x 20-period average
# Exits when price closes opposite Camarilla level (H4/L4)
# Position size 0.20 to limit drawdown and enable discrete sizing
# Target: 15-37 trades/year per symbol (60-150 total over 4 years) to minimize fee drag
# Works in both bull/bear: Camarilla provides structure, 4h trend filter avoids counter-trend trades

name = "1h_4h_1d_camarilla_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        pivot = (phigh + plow + pclose) / 3
        range_val = phigh - plow
        
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA(21)
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(df_4h), np.nan)
    multiplier = 2 / (21 + 1)
    ema_4h[20] = close_4h[:21].mean()  # SMA for first value
    for i in range(21, len(df_4h)):
        ema_4h[i] = (close_4h[i] - ema_4h[i-1]) * multiplier + ema_4h[i-1]
    
    # Align HTF data to 1h timeframe (only use completed bars)
    camarilla_h3_1h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_1h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_1h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_1h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: 20-period average on 1h
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_1h[i]) or 
            np.isnan(camarilla_l3_1h[i]) or 
            np.isnan(camarilla_h4_1h[i]) or 
            np.isnan(camarilla_l4_1h[i]) or 
            np.isnan(ema_4h_1h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Camarilla L4
            if close[i] <= camarilla_l4_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Camarilla H4
            if close[i] >= camarilla_h4_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price closes above 1d Camarilla H3 with volume confirmation and 4h uptrend
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > camarilla_h3_1h[i] and 
                vol_ratio > 1.5 and
                close[i] > ema_4h_1h[i]):
                position = 1
                signals[i] = 0.20
            # Enter short: price closes below 1d Camarilla L3 with volume confirmation and 4h downtrend
            elif (close[i] < camarilla_l3_1h[i] and 
                  vol_ratio > 1.5 and
                  close[i] < ema_4h_1h[i]):
                position = -1
                signals[i] = -0.20
    
    return signals