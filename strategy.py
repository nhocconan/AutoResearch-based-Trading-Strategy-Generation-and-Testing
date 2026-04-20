#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime (daily trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 21-period EMA on daily close for trend filter
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 10-day ATR on daily for volatility filter
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_10d_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_10d_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10d_1d)
    
    # Calculate 4h price channel (Donchian 20)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for stop loss
    tr1_4h = high_4h[1:] - low_4h[1:]
    tr2_4h = np.abs(high_4h[1:] - close_4h[:-1])
    tr3_4h = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in any indicator
        if np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_10d_1d_aligned[i]) or \
           np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        price = close_4h[i]
        
        # Daily trend filter: price above/below daily EMA21
        trend_up = price > ema_21_1d_aligned[i]
        trend_down = price < ema_21_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter_enhanced = atr_4h[i] > 0.5 * atr_10d_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above channel + uptrend + volume + volatility
            if price > upper_channel and trend_up and vol_filter and vol_filter_enhanced:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below channel + downtrend + volume + volatility
            elif price < lower_channel and trend_down and vol_filter and vol_filter_enhanced:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below channel OR trend reverses
            if price < lower_channel or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above channel OR trend reverses
            if price > upper_channel or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_DailyEMA21_VolVolFilter_Session"
timeframe = "4h"
leverage = 1.0