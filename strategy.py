# 4h_Combo_Filter_Strategy
# Hypothesis: Combines 4h price momentum with 1d trend filter and volume spike to capture strong trends in both bull and bear markets.
# Uses RSI for momentum, EMA for trend, and volume confirmation to reduce false signals.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.

name = "4h_Combo_Filter_Strategy"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1h data for momentum filter (RSI)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate RSI on 1h closes (14-period)
    close_1h = df_1h['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi)

    # Calculate 50-period EMA on 1d closes for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)

    # Volume confirmation: 20-period average on 4h data
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        rsi_val = rsi_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any required data is NaN
        if (np.isnan(rsi_val) or np.isnan(ema_trend) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 55 (bullish momentum), price above EMA50 (uptrend), volume surge
            if (rsi_val > 55 and 
                close[i] > ema_trend and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45 (bearish momentum), price below EMA50 (downtrend), volume surge
            elif (rsi_val < 45 and 
                  close[i] < ema_trend and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 (momentum fading) or price below EMA50 (trend change)
            if (rsi_val < 40 or close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 60 (momentum fading) or price above EMA50 (trend change)
            if (rsi_val > 60 or close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals