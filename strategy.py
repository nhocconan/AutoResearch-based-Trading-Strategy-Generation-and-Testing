#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1W/1D Mean Reversion with Volume Confirmation
# Uses weekly RSI(14) for extreme conditions and daily EMA(34) for trend filter
# Goes long when weekly RSI < 30 and daily EMA is rising (oversold bounce)
# Goes short when weekly RSI > 70 and daily EMA is falling (overbought rejection)
# Volume confirmation via 20-period volume spike to ensure participation
# Target: 12-37 trades/year (50-150 total over 4 years) with strict entry conditions

name = "12h_WeeklyRSI_DailyEMA_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Weekly RSI(14) for mean reversion signals
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w[:13] = np.nan  # Not enough data for first 13 periods
    
    # Daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_rising = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    ema_rising = np.concatenate([[False], ema_rising])  # Align with 1d index
    
    # Volume confirmation: 20-period volume spike (1.5x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    # Align weekly RSI and daily EMA to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Weekly oversold + Daily uptrend + Volume spike
            if (rsi_aligned[i] < 30 and  # Weekly RSI oversold
                ema_rising_aligned[i] > 0.5 and  # Daily EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Weekly overbought + Daily downtrend + Volume spike
            elif (rsi_aligned[i] > 70 and  # Weekly RSI overbought
                  ema_rising_aligned[i] <= 0.5 and  # Daily EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Weekly RSI returns to neutral or trend breaks
            if (rsi_aligned[i] >= 50 or  # RSI back to neutral
                ema_rising_aligned[i] <= 0.5):  # Daily trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Weekly RSI returns to neutral or trend breaks
            if (rsi_aligned[i] <= 50 or  # RSI back to neutral
                ema_rising_aligned[i] > 0.5):  # Daily trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals