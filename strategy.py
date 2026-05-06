#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d RSI extremes with volume confirmation and 1w trend filter
# - Uses 1d RSI(14) < 30 for oversold/long signals and > 70 for overbought/short signals
# - Requires volume > 1.5x 20-period average for confirmation
# - Uses 1w EMA(50) slope to filter trades in direction of higher timeframe trend
# - Designed to capture mean reversion in strong trends with institutional volume
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_1dRSI_1wEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d RSI (14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period):
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        if len(gain) < period:
            return np.full_like(gain, 50.0)
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = wilders_rsi(gain, loss, 14)
    
    # Calculate 1w EMA(50) slope for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = np.diff(ema_50, prepend=ema_50[0])
    
    # Align 1d indicators to 4h timeframe
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Align 1w EMA slope to 4h timeframe
    ema_slope_4h = align_htf_to_ltf(prices, df_1w, ema_slope)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(rsi_1d_4h[i]) or np.isnan(ema_slope_4h[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + volume confirmation + uptrend (EMA slope > 0)
            if rsi_1d_4h[i] < 30 and volume_confirm[i] and ema_slope_4h[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + volume confirmation + downtrend (EMA slope < 0)
            elif rsi_1d_4h[i] > 70 and volume_confirm[i] and ema_slope_4h[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or trend changes
            if rsi_1d_4h[i] > 50 or ema_slope_4h[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or trend changes
            if rsi_1d_4h[i] < 50 or ema_slope_4h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals