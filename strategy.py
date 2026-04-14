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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) - Volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily volatility filter (ATR > 1.5% of price)
    vol_filter_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and close_1d[i] > 0:
            vol_filter_1d[i] = atr_1d[i] / close_1d[i] > 0.015
        else:
            vol_filter_1d[i] = False
    
    # Calculate daily RSI (14-period) - Momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(len(df_1d), np.nan)
    avg_loss = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(df_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(avg_loss[i]) and avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_1d[i] = 100 - (100 / (1 + rs))
        elif not np.isnan(avg_gain[i]) and avg_loss[i] == 0:
            rsi_1d[i] = 100.0
    
    # Calculate daily ADX (14-period) - Trend strength filter
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    tr_14 = np.zeros(len(df_1d))
    tr_14[0] = tr[0]
    for i in range(1, len(df_1d)):
        tr_14[i] = tr[i]
    
    atr_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_14[13] = np.mean(tr_14[:14])
        for i in range(14, len(df_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr_14[i]) / 14
    
    plus_di = np.full(len(df_1d), np.nan)
    minus_di = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        for i in range(13, len(df_1d)):
            if atr_14[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-13:i+1]) / atr_14[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-13:i+1]) / atr_14[i])
    
    dx = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        for i in range(13, len(df_1d)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:  # Need 14+14 for smoothing
        adx_1d[26] = np.mean(dx[13:27])
        for i in range(27, len(df_1d)):
            adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align indicators to 1h timeframe (primary timeframe)
    atr_1h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_filter_1h = align_htf_to_ltf(prices, df_1d, vol_filter_1d.astype(float))
    rsi_1h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    adx_1h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Pre-compute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(rsi_1h[i]) or
            np.isnan(adx_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: Only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1.5% of price)
        if vol_filter_1h[i] < 0.5:
            signals[i] = 0.0
            continue
        
        # Require strong trend (ADX > 25)
        if adx_1h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Calculate daily pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels (R3/S3)
        r3 = prev_close + (prev_range * 1.1 / 4)
        s3 = prev_close - (prev_range * 1.1 / 4)
        
        # Align to 1h timeframe
        r3_1h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
        s3_1h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
        
        if position == 0:
            # Long: Price breaks above 1h Donchian high AND above S3 AND RSI < 70 AND ADX > 25
            if close[i] > donch_high[i] and close[i] > s3_1h and rsi_1h[i] < 70:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 1h Donchian low AND below R3 AND RSI > 30 AND ADX > 25
            elif close[i] < donch_low[i] and close[i] < r3_1h and rsi_1h[i] > 30:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 1h Donchian low OR below S3 OR RSI > 70
            if close[i] < donch_low[i] or close[i] < s3_1h or rsi_1h[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 1h Donchian high OR above R3 OR RSI < 30
            if close[i] > donch_high[i] or close[i] > r3_1h or rsi_1h[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d_ADX_RSI_Camarilla_Filter"
timeframe = "1h"
leverage = 1.0