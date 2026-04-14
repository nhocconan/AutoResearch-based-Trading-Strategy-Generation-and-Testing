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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily EMA (20-period)
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily ADX (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = tr
    plus_di_14 = np.full(len(df_1d), np.nan)
    minus_di_14 = np.full(len(df_1d), np.nan)
    dx_14 = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 14:
        # Smooth +DM, -DM, TR
        plus_dm_smooth = np.full(len(df_1d), np.nan)
        minus_dm_smooth = np.full(len(df_1d), np.nan)
        tr_smooth = np.full(len(df_1d), np.nan)
        
        plus_dm_smooth[13] = np.sum(plus_dm[1:15])
        minus_dm_smooth[13] = np.sum(minus_dm[1:15])
        tr_smooth[13] = np.sum(tr[1:15])
        
        for i in range(14, len(df_1d)):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
        
        plus_di_14 = 100 * plus_dm_smooth / tr_smooth
        minus_di_14 = 100 * minus_dm_smooth / tr_smooth
        dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    
    adx_14 = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 27:  # Need 14 + 14 for smoothing
        adx_14[26] = np.mean(dx_14[14:28])
        for i in range(27, len(df_1d)):
            adx_14[i] = (adx_14[i-1] * 13 + dx_14[i]) / 14
    
    # Align indicators to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_20_4h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 4-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_20_4h[i]) or
            np.isnan(rsi_4h[i]) or
            np.isnan(volume_ma[i]) or
            np.isnan(adx_4h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_4h[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 80% of 20-period MA)
        if volume[i] < 0.8 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip low trend strength (ADX < 25)
        if adx_4h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Calculate pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels (R4/S4)
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        
        # Align to 4h timeframe
        r4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r4))[i]
        s4_4h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s4))[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high AND above S4 AND ADX > 25 AND RSI > 50 AND price above daily EMA20
            if close[i] > donch_high[i] and close[i] > s4_4h and adx_4h[i] > 25 and rsi_4h[i] > 50 and close[i] > ema_20_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low AND below R4 AND ADX > 25 AND RSI < 50 AND price below daily EMA20
            elif close[i] < donch_low[i] and close[i] < r4_4h and adx_4h[i] > 25 and rsi_4h[i] < 50 and close[i] < ema_20_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 4h Donchian low OR below S4 OR ADX < 20 OR RSI < 50 OR price below daily EMA20
            if close[i] < donch_low[i] or close[i] < s4_4h or adx_4h[i] < 20 or rsi_4h[i] < 50 or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 4h Donchian high OR above R4 OR ADX < 20 OR RSI > 50 OR price above daily EMA20
            if close[i] > donch_high[i] or close[i] > r4_4h or adx_4h[i] < 20 or rsi_4h[i] > 50 or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_R4S4_RSI50_EMA20_ADX_Filter_Volume"
timeframe = "4h"
leverage = 1.0