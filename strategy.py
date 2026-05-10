#!/usr/bin/env python3
# 4h_RSI20_Trend_1d_Engulfing
# Hypothesis: RSI(20) < 40 with bullish daily trend for longs, RSI(20) > 60 with bearish daily trend for shorts.
# Uses engulfing candle patterns for entry timing and volume confirmation.
# Designed to work in both bull and bear markets by aligning with daily trend.
# Targets 20-40 trades/year to minimize fee drag.

name = "4h_RSI20_Trend_1d_Engulfing"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # RSI(20) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss = loss.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > high[1:]) & (open_price < low[1:])  # Will be adjusted in loop
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulfing = (close < open_price) & (open_price > close) & \
                        (open_price > high[1:]) & (close < low[1:])  # Will be adjusted in loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 20) + 1  # RSI warmup + daily EMA + volume MA + engulfing lookback
    
    for i in range(start_idx, n):
        # Shift arrays for engulfing calculation (previous bar)
        if i < 1:
            continue
            
        # Skip if any critical values are NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Engulfing patterns (using previous bar)
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i-1] > close[i-1]) and \
                         (close[i] >= open_price[i-1]) and (open_price[i] <= close[i-1])
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i-1] < close[i-1]) and \
                         (open_price[i] >= close[i-1]) and (close[i] <= open_price[i-1])
        
        if position == 0:
            # Long entry: RSI < 40 (oversold) + bullish engulfing + uptrend + volume spike
            if rsi_values[i] < 40 and bullish_engulf and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 60 (overbought) + bearish engulfing + downtrend + volume spike
            elif rsi_values[i] > 60 and bearish_engulf and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or trend reversal
            if rsi_values[i] > 60 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or trend reversal
            if rsi_values[i] < 40 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals