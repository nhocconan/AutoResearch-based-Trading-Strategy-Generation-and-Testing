#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI + 4h/1d trend filter with session and volume confirmation.
# Uses 4h EMA20 for trend direction, 1h RSI(14) for momentum, and volume spike for confirmation.
# Long when 4h trend up, RSI > 55, volume > 1.5x average. Short when 4h trend down, RSI < 45, volume > 1.5x average.
# Includes session filter (08-20 UTC) to avoid low-volatility periods.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "1h_RSI_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA(20) for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h_up = ema_20_4h > np.roll(ema_20_4h, 1)
    trend_4h_up = np.where(np.isnan(trend_4h_up), False, trend_4h_up)
    
    # 1d EMA(20) for trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1d_up = ema_20_1d > np.roll(ema_20_1d, 1)
    trend_1d_up = np.where(np.isnan(trend_1d_up), False, trend_1d_up)
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    # Align 1d trend to 1h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    # 1h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_1d_up_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or np.isnan(in_session[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h trend up, 1d trend up, RSI > 55, volume spike
            if (trend_4h_up_aligned[i] and trend_1d_up_aligned[i] and
                rsi[i] > 55 and vol_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: 4h trend down, 1d trend down, RSI < 45, volume spike
            elif (not trend_4h_up_aligned[i] and not trend_1d_up_aligned[i] and
                  rsi[i] < 45 and vol_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend break or RSI < 40
            if (not trend_4h_up_aligned[i] or not trend_1d_up_aligned[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend break or RSI > 60
            if (trend_4h_up_aligned[i] or trend_1d_up_aligned[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals