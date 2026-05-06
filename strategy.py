#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) with 1d EMA(50) trend filter and volume confirmation
# Uses 4h RSI for mean-reversion entries (RSI<30 long, RSI>70 short)
# Requires price to be above/below 1d EMA(50) for trend alignment
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: RSI mean reversion works in ranging markets, EMA filter avoids counter-trend trades

name = "4h_RSI14_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 14 or len(df_1d) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 4h timeframe
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Calculate EMA(50) on 1d timeframe
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) AND price above 1d EMA(50) (uptrend) AND volume confirmation
            if (rsi_4h_aligned[i] < 30 and close[i] > ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 70 (overbought) AND price below 1d EMA(50) (downtrend) AND volume confirmation
            elif (rsi_4h_aligned[i] > 70 and close[i] < ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (neutral) or price below 1d EMA(50)
            if rsi_4h_aligned[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (neutral) or price above 1d EMA(50)
            if rsi_4h_aligned[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals