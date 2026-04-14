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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA (20-period) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    low_close = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(close_1w), np.nan)
    avg_loss = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily timeframe
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    atr_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate daily volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d[i]) or
            np.isnan(atr_1d[i]) or
            np.isnan(rsi_1d[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_1d[i] / close[i] < 0.005:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 80% of 20-period MA)
        if volume[i] < 0.8 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price must be above weekly EMA20 for long, below for short
        long_trend = close[i] > ema_20_1d[i]
        short_trend = close[i] < ema_20_1d[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_ok = 30 <= rsi_1d[i] <= 70
        
        if position == 0:
            # Long: Price breaks above daily Donchian high + trend + momentum
            if close[i] > donch_high[i] and long_trend and rsi_ok:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below daily Donchian low + trend + momentum
            elif close[i] < donch_low[i] and short_trend and rsi_ok:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below daily Donchian low OR trend breaks
            if close[i] < donch_low[i] or not long_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above daily Donchian high OR trend breaks
            if close[i] > donch_high[i] or not short_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_EMA20_RSI_Filter"
timeframe = "1d"
leverage = 1.0