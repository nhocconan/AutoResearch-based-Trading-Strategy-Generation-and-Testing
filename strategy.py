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
    
    # Get weekly data for trend filter and volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR for volatility filter (14-period)
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_14_1w = np.full(len(tr_1w), np.nan)
    for i in range(13, len(tr_1w)):
        atr_14_1w[i] = np.nanmean(tr_1w[i-13:i+1])
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Weekly EMA for trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily close for Donchian breakout calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 20-period Donchian channels on daily
    upper_20_1d = np.full(len(close_1d), np.nan)
    lower_20_1d = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        upper_20_1d[i] = np.max(close_1d[i-19:i+1])
        lower_20_1d[i] = np.min(close_1d[i-19:i+1])
    
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Volume filter: volume > 1.5 x 20-day average
    vol_ma_20d = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly ATR (14), weekly EMA (21), daily Donchian (20), volume MA (20)
    start_idx = max(14, 21, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_1w_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(upper_20_1d_aligned[i]) or np.isnan(lower_20_1d_aligned[i]) or
            np.isnan(vol_ma_20d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20d[i]
        atr_1w = atr_14_1w_aligned[i]
        ema_21w = ema_21_1w_aligned[i]
        upper_donchian = upper_20_1d_aligned[i]
        lower_donchian = lower_20_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_1w > np.nanmedian(atr_14_1w_aligned[max(0, i-50):i+1]) if not np.isnan(np.nanmedian(atr_14_1w_aligned[max(0, i-50):i+1])) else False
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_21w
        bearish_trend = price < ema_21w
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + volatility + bullish weekly trend
            if price > upper_donchian and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly Donchian lower + volatility + bearish weekly trend
            elif price < lower_donchian and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower or trend turns bearish
            if price < lower_donchian or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper or trend turns bullish
            if price > upper_donchian or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_21wEMA_VolatilityFilter"
timeframe = "1d"
leverage = 1.0