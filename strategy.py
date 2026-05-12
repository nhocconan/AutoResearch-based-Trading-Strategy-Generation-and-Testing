#!/usr/bin/env python3
# 6h_Momentum_Pullback_12hTrend
# Hypothesis: Momentum pullback strategy for 6h timeframe. Uses 12h EMA trend filter,
# 6h RSI pullback to EMA, and volume confirmation. Enters long in uptrend when RSI
# pulls back from overbought to neutral, short in downtrend when RSI pulls back
# from oversold to neutral. Designed for low frequency (15-35 trades/year) to avoid
# fee drag. Works in bull markets by catching uptrend continuations and in bear
# markets by catching downtrend continuations after pullbacks.

name = "6h_Momentum_Pullback_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h RSI
    rsi_6h = rsi(close, 14)
    
    # Calculate 6h EMA20 for pullback target
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # RSI conditions for pullback entry
        rsi_value = rsi_6h[i]
        rsi_overbought = rsi_value > 70
        rsi_oversold = rsi_value < 30
        rsi_neutral = (rsi_value >= 40) & (rsi_value <= 60)
        
        # Price near EMA20 (within 1%)
        price_near_ema = abs(close[i] - ema_20_6h[i]) / ema_20_6h[i] < 0.01
        
        if position == 0:
            # LONG: Uptrend + RSI pullback from overbought to neutral near EMA
            if trend_up and vol_ok and rsi_neutral and price_near_ema and rsi_overbought:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + RSI pullback from oversold to neutral near EMA
            elif trend_down and vol_ok and rsi_neutral and price_near_ema and rsi_oversold:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend failure or RSI re-enters overbought
            if not trend_up or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend failure or RSI re-enters oversold
            if not trend_down or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals