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
    
    # Load 12h data for trend (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h ADX(14) for trend strength
    if len(df_12h) >= 14:
        tr_12h = np.zeros(len(df_12h))
        tr_12h[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(df_12h)):
            tr_12h[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
        
        plus_dm_12h = np.zeros(len(df_12h))
        minus_dm_12h = np.zeros(len(df_12h))
        for i in range(1, len(df_12h)):
            up_move = high_12h[i] - high_12h[i-1]
            down_move = low_12h[i-1] - low_12h[i]
            plus_dm_12h[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm_12h[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        atr_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 14:
            atr_12h[13] = np.mean(tr_12h[:14])
            for i in range(14, len(df_12h)):
                atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
        
        plus_di_12h = np.full(len(df_12h), np.nan)
        minus_di_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 14 and not np.all(np.isnan(atr_12h)):
            for i in range(13, len(df_12h)):
                if atr_12h[i] > 0:
                    plus_di_12h[i] = 100 * np.mean(plus_dm_12h[i-13:i+1]) / atr_12h[i]
                    minus_di_12h[i] = 100 * np.mean(minus_dm_12h[i-13:i+1]) / atr_12h[i]
        
        dx_12h = np.full(len(df_12h), np.nan)
        for i in range(13, len(df_12h)):
            if plus_di_12h[i] + minus_di_12h[i] > 0:
                dx_12h[i] = 100 * abs(plus_di_12h[i] - minus_di_12h[i]) / (plus_di_12h[i] + minus_di_12h[i])
        
        adx_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 27:
            adx_12h[26] = np.mean(dx_12h[13:27])
            for i in range(27, len(df_12h)):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx_12h[i]) / 14
    else:
        adx_12h = np.full(len(df_12h), np.nan)
    
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Load 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h EMA(20) for entry filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 12h EMA50 when ADX > 20
        if adx_12h_aligned[i] > 20:
            trend_up = close[i] > ema_50_12h_aligned[i]
            trend_down = close[i] < ema_50_12h_aligned[i]
        else:
            trend_up = False
            trend_down = False
        
        # Volatility filter: require sufficient volatility
        if atr_1d_aligned[i] < 0.005 * close[i]:  # Less than 0.5% ATR
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above EMA20 and in uptrend
            if close[i] > ema_20[i] and trend_up:
                position = 1
                signals[i] = position_size
            # Short: Price below EMA20 and in downtrend
            elif close[i] < ema_20[i] and trend_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls below EMA20
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises above EMA20
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_EMA50_ADX20_EMA20_Trend"
timeframe = "4h"
leverage = 1.0