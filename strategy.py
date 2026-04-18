#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and RSI
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) channels on daily
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(close_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # RSI(14) on daily
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period + 1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.full_like(close, np.nan)
        rsi = np.full_like(close, np.nan)
        valid = avg_loss != 0
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        rsi[valid] = 100 - (100 / (1 + rs[valid]))
        rsi[avg_loss == 0] = 100
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to daily timeframe
    upper_daily = align_htf_to_ltf(prices, df_1d, upper)
    lower_daily = align_htf_to_ltf(prices, df_1d, lower)
    rsi_daily = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_1w_daily = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_daily[i]) or np.isnan(lower_daily[i]) or 
            np.isnan(rsi_daily[i]) or np.isnan(ema_1w_daily[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above weekly EMA
        uptrend = close[i] > ema_1w_daily[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with RSI < 70 and volume
            if close[i] > upper_daily[i] and rsi_daily[i] < 70 and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with RSI > 30 and volume
            elif close[i] < lower_daily[i] and rsi_daily[i] > 30 and vol_confirm and not uptrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR RSI > 70
            if close[i] < lower_daily[i] or rsi_daily[i] > 70:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR RSI < 30
            if close[i] > upper_daily[i] or rsi_daily[i] < 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_RSI_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0