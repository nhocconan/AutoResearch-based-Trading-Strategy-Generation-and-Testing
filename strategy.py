#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h EMA trend filter and volume spike.
# Uses 4h EMA50 for trend direction, RSI(14) for overbought/oversold signals,
# and volume surge for confirmation. Designed to work in both bull (pullbacks in uptrend)
# and bear (bounces in downtrend). Target: 15-37 trades/year to avoid fee drag.
name = "1h_RSI14_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA for 4h timeframe
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) for 1h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.8x 24-period EMA (moderate threshold)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_confirm = volume > (1.8 * vol_ema24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA and RSI stability
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ema24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume spike
            if (rsi[i] < 30 and price > ema_50_4h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume spike
            elif (rsi[i] > 70 and price < ema_50_4h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (overbought) or price below 4h EMA50
            if rsi[i] > 50 or price < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 (oversold) or price above 4h EMA50
            if rsi[i] < 50 or price > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals