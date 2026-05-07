#!/usr/bin/env python3
# 1h_RSI_MeanReversion_4hTrendFilter_VolumeConfirm
# Hypothesis: RSI mean reversion on 1h with 4h trend filter and volume confirmation
# captures mean-reverting moves within the dominant trend, reducing false signals.
# Works in both bull and bear markets by aligning with higher timeframe trend.
# Target: 15-35 trades/year to minimize fee drag while capturing high-probability reversals.

name = "1h_RSI_MeanReversion_4hTrendFilter_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter and RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend filter
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_trend = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h RSI(14) for entry signals
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    # Calculate 1h volume moving average for confirmation
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi_4h_trend[i]) or np.isnan(rsi_1h[i]) or 
            np.isnan(vol_ma_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: RSI oversold (<30) on 1h, 4h trend bullish (RSI>50), volume above average
            if rsi_1h[i] < 30 and rsi_4h_trend[i] > 50 and volume[i] > vol_ma_20_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) on 1h, 4h trend bearish (RSI<50), volume above average
            elif rsi_1h[i] > 70 and rsi_4h_trend[i] < 50 and volume[i] > vol_ma_20_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (40-60) or trend weakens
            if rsi_1h[i] >= 40 or rsi_4h_trend[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI returns to neutral (40-60) or trend weakens
            if rsi_1h[i] <= 60 or rsi_4h_trend[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals