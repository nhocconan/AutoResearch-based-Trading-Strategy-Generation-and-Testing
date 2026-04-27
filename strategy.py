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
    
    # Get 12h data for ATR and price structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR(14) on 12h for volatility filter and stop sizing
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for price structure (Donchian channel breakout)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channel (20-period) on 4h data
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume filter: volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h ATR (14), 1d EMA (50), 4h Donchian (20), volume MA (20)
    start_idx = max(14, 50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        atr_val = atr_14_aligned[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility chop)
        vol_filter = atr_val > 0.005 * price
        
        # Volume filter: volume > 1.5x average
        vol_spike = vol_now > 1.5 * vol_avg
        
        # Trend filter: price vs 1d EMA50
        bullish_trend = price > ema_trend_1d
        bearish_trend = price < ema_trend_1d
        
        if position == 0:
            # Long: price breaks above Donchian high + bullish trend + volatility + volume
            if price > donch_high and bullish_trend and vol_filter and vol_spike:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + bearish trend + volatility + volume
            elif price < donch_low and bearish_trend and vol_filter and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: ATR-based trailing stop or trend reversal
            if price <= donch_high - 2.0 * atr_val or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: ATR-based trailing stop or trend reversal
            if price >= donch_low + 2.0 * atr_val or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ATR_Volume_Donchian_Breakout_4h_1dEMA50"
timeframe = "12h"
leverage = 1.0