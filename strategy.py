#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and session filter
# Long when: 1h RSI crosses above 50 AND 4h close > 1d EMA200 AND 1h volume > 1.5x avg volume
# Short when: 1h RSI crosses below 50 AND 4h close < 1d EMA200 AND 1h volume > 1.5x avg volume
# Exit when RSI crosses back to 50 or trend reverses
# Session filter: 08-20 UTC only
# Target: 15-37 trades/year on 1h (60-150 total over 4 years)

name = "1h_4h_1d_rsi_trend_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute hour for session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h close for trend filter
    close_4h = df_4h['close'].values
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate 14-period RSI on 1h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Wait for EMA200 to be valid
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(close_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filters
        trend_1d_up = close[i] > ema_200_1d_aligned[i]
        trend_1d_down = close[i] < ema_200_1d_aligned[i]
        trend_4h_up = close_4h_aligned[i] > ema_200_1d_aligned[i]
        trend_4h_down = close_4h_aligned[i] < ema_200_1d_aligned[i]
        
        # RSI crossover signals
        rsi_cross_up = rsi[i-1] < 50 and rsi[i] >= 50
        rsi_cross_down = rsi[i-1] > 50 and rsi[i] <= 50
        
        # Entry conditions
        long_entry = rsi_cross_up and volume_filter and trend_1d_up and trend_4h_up and in_session
        short_entry = rsi_cross_down and volume_filter and trend_1d_down and trend_4h_down and in_session
        
        # Exit conditions
        long_exit = rsi_cross_down or not (trend_1d_up and trend_4h_up) or not in_session
        short_exit = rsi_cross_up or not (trend_1d_down and trend_4h_down) or not in_session
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals