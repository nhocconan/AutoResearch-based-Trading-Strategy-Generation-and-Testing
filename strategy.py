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
    
    # Get 1d data for EMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 14-period RSI on 1d
    def calculate_rsi(close, period=14):
        if len(close) < period + 1:
            return np.full_like(close, np.nan)
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Get 4h data for volume
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period volume average on 4h
    vol_ma_4h = np.full_like(volume_4h, np.nan)
    vol_period = 20
    
    if len(volume_4h) >= vol_period:
        for i in range(vol_period, len(volume_4h)):
            vol_ma_4h[i] = np.mean(volume_4h[i-vol_period:i])
    
    # Align all data to 1h timeframe
    ema_1d_1h = align_htf_to_ltf(prices, df_1d, ema_1d)
    rsi_1d_1h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_4h_1h = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(ema_1d_1h[i]) or np.isnan(rsi_1d_1h[i]) or 
            np.isnan(vol_ma_4h_1h[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (4h)
        vol_confirm = volume[i] > 1.5 * vol_ma_4h_1h[i]
        
        # Trend filter: price above/below 50-day EMA
        uptrend = close[i] > ema_1d_1h[i]
        downtrend = close[i] < ema_1d_1h[i]
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_1d_1h[i] < 70
        rsi_not_oversold = rsi_1d_1h[i] > 30
        
        if position == 0:
            # Long: price above EMA, RSI not overbought, volume confirmation
            if uptrend and rsi_not_overbought and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA, RSI not oversold, volume confirmation
            elif downtrend and rsi_not_oversold and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA OR RSI overbought
            if not uptrend or rsi_1d_1h[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above EMA OR RSI oversold
            if not downtrend or rsi_1d_1h[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_RSI14_Volume_Session"
timeframe = "1h"
leverage = 1.0