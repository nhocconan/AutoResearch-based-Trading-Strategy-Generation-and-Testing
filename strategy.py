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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 14-day RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    rsi = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing
    for i in range(1, len(close_1d)):
        if i < 14:
            if i == 1:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Calculate 50-week EMA on weekly for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Calculate 14-day ATR on daily
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align all data to daily timeframe (primary)
    upper_channel_daily = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_daily = align_htf_to_ltf(prices, df_1d, lower_channel)
    rsi_daily = align_htf_to_ltf(prices, df_1d, rsi)
    ema_1w_daily = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1d_daily = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 14, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_daily[i]) or np.isnan(lower_channel_daily[i]) or 
            np.isnan(rsi_daily[i]) or np.isnan(ema_1w_daily[i]) or 
            np.isnan(atr_1d_daily[i])):
            signals[i] = 0.0
            continue
        
        # RSI filter: avoid extreme levels (30-70 range for mean reversion)
        rsi_filter = (rsi_daily[i] >= 30) & (rsi_daily[i] <= 70)
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_daily[i]
        downtrend = close[i] < ema_1w_daily[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_daily[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        if position == 0:
            # Long: price breaks below lower Donchian channel (mean reversion) in uptrend
            if close[i] < lower_channel_daily[i] and uptrend and rsi_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above upper Donchian channel (mean reversion) in downtrend
            elif close[i] > upper_channel_daily[i] and downtrend and rsi_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above upper Donchian channel OR RSI overbought
            if close[i] > upper_channel_daily[i] or rsi_daily[i] > 70:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below lower Donchian channel OR RSI oversold
            if close[i] < lower_channel_daily[i] or rsi_daily[i] < 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_RSI_MeanReversion_1wEMA_v1"
timeframe = "1d"
leverage = 1.0