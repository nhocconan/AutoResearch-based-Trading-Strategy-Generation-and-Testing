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
    
    # Get weekly data for trend and volatility filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 10-week EMA for trend filter
    ema10_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[:10])
        for i in range(10, len(close_1w)):
            ema10_1w[i] = (close_1w[i] * (2 / (10 + 1)) + ema10_1w[i-1] * (1 - (2 / (10 + 1))))
    
    # Calculate 20-week ATR for volatility filter
    tr_1w = np.zeros(len(high_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        hl = high_1w[i] - low_1w[i]
        hc = abs(high_1w[i] - close_1w[i-1])
        lc = abs(low_1w[i] - close_1w[i-1])
        tr_1w[i] = max(hl, hc, lc)
    
    atr20_1w = np.full(len(tr_1w), np.nan)
    for i in range(19, len(tr_1w)):
        if i == 19:
            atr20_1w[i] = np.mean(tr_1w[:20])
        else:
            atr20_1w[i] = (atr20_1w[i-1] * 19 + tr_1w[i]) / 20
    
    # Align weekly indicators to daily timeframe
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    atr20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr20_1w)
    
    # Calculate daily Donchian channels (20-day)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily volume moving average (20-day)
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(atr20_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ma20_val = vol_ma20[i]
        vol_current = volume[i]
        ema_trend = ema10_1w_aligned[i]
        atr_vol = atr20_1w_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = vol_current > vol_ma20_val * 0.8
        
        if position == 0:
            # Long: Price breaks above Donchian high with weekly uptrend and volume
            if (price > donch_high_val and price > ema_trend and vol_filter):
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with weekly downtrend and volume
            elif (price < donch_low_val and price < ema_trend and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below Donchian low or trend fails
            if price < donch_low_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price breaks above Donchian high or trend fails
            if price > donch_high_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1D_Donchian_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0