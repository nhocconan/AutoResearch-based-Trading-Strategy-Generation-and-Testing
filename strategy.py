#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Trend + Daily Pullback Strategy
# Uses weekly EMA trend filter to capture major trends, entering on daily pullbacks
# with volume confirmation. Works in both bull and bear markets by following the
# weekly trend. Weekly timeframe reduces noise and false signals, while daily
# entries provide better timing. Target: 10-20 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily data for entry signals and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA(21) trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (21 + 1)
    ema_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily RSI(14) for pullback identification
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close_1d))
    avg_loss = np.zeros(len(close_1d))
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else np.mean(gain) if len(gain) > 0 else 0
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else np.mean(loss) if len(loss) > 0 else 0
    
    for i in range(1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily average volume (20-period)
    avg_volume_1d = np.zeros(len(close_1d))
    for i in range(20, len(close_1d)):
        avg_volume_1d[i] = np.mean(volume[i-20:i])
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        weekly_ema = ema_1w_aligned[i]
        daily_rsi = rsi_1d_aligned[i]
        avg_vol = avg_volume_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: weekly uptrend + daily pullback (RSI < 40) + volume
            if (price > weekly_ema and  # Above weekly EMA = uptrend
                daily_rsi < 40 and      # Oversold on daily
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: weekly downtrend + daily bounce (RSI > 60) + volume
            elif (price < weekly_ema and   # Below weekly EMA = downtrend
                  daily_rsi > 60 and       # Overbought on daily
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or RSI overbought
            if (price < weekly_ema or
                daily_rsi > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or RSI oversold
            if (price > weekly_ema or
                daily_rsi < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "weekly_trend_daily_pullback_v1"
timeframe = "1d"
leverage = 1.0