# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h/1d trend filter
# - Long when RSI(14) < 30 and price > 4h EMA(50) and price > 1d EMA(200)
# - Short when RSI(14) > 70 and price < 4h EMA(50) and price < 1d EMA(200)
# - Exit when RSI returns to neutral (40-60 range) or opposite extreme
# - Uses 4h/1d for trend direction, 1h for RSI timing
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-37 trades/year (~60-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 4h data for EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for EMA(200) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(np.concatenate([[np.nan], close]))
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if NaN in indicators
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend conditions
        uptrend = (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_200_1d_aligned[i])
        downtrend = (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_200_1d_aligned[i])
        
        if position == 0:
            # Long entry: RSI oversold + uptrend
            if rsi[i] < 30 and uptrend:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought + downtrend
            elif rsi[i] > 70 and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or turns bearish
            if rsi[i] > 40 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral or turns bullish
            if rsi[i] < 60 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4h1dEMAFilter_Session"
timeframe = "1h"
leverage = 1.0