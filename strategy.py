#!/usr/bin/env python3
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
    
    # Get 4h data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for longer-term context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA(50) for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h ATR(14) for volatility filter
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h[:-1]),
            np.abs(low_4h[1:] - close_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 4h Donchian(20) for breakout levels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high_20 = rolling_max(high_4h, 20)
    donch_low_20 = rolling_min(low_4h, 20)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # Volume filter: 4h volume > 1.5 x 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need 4h EMA(50), 1d EMA(200), ATR(14), Donchian(20), volume MA
    start_idx = max(50, 200, 14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_4h_aligned[i]
        vol_filter = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Trend filters
        bullish_4h = price > ema_50_4h_aligned[i]
        bullish_1d = price > ema_200_1d_aligned[i]
        
        # Breakout conditions with volatility-adjusted thresholds
        upper_break = donch_high_20_aligned[i] + 0.5 * atr
        lower_break = donch_low_20_aligned[i] - 0.5 * atr
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and bullish trend
            if price > upper_break and vol_filter and bullish_4h and bullish_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with volume and bearish trend
            elif price < lower_break and vol_filter and not bullish_4h and not bullish_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend turns bearish
            if price < donch_low_20_aligned[i] or not (bullish_4h and bullish_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend turns bullish
            if price > donch_high_20_aligned[i] or (bullish_4h and bullish_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_DonchianBreakout_4h1dEMA_VolumeFilter"
timeframe = "1h"
leverage = 1.0