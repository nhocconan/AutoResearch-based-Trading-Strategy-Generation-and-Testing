#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) with 4h/1d trend filter and volume confirmation.
# RSI identifies overbought/oversold conditions while 4h trend (EMA50) and 1d trend (EMA200) filter trades.
# Volume confirmation ensures institutional participation. Works in both bull/bear by taking longs in uptrend,
# shorts in downtrend. Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter and entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = np.zeros(len(close_4h))
    ema_multiplier = 2 / (50 + 1)
    ema50_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        ema50_4h[i] = (close_4h[i] - ema50_4h[i-1]) * ema_multiplier + ema50_4h[i-1]
    
    # Calculate 1d EMA(200) for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema200_1d = np.zeros(len(close_1d))
    ema_multiplier200 = 2 / (200 + 1)
    ema200_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema200_1d[i] = (close_1d[i] - ema200_1d[i-1]) * ema_multiplier200 + ema200_1d[i-1]
    
    # Align indicators to 1h timeframe
    rsi_aligned = rsi  # Already on 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate average volume (24-period = 1 day) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi_aligned[i]
        ema40_trend = ema50_4h_aligned[i]
        ema1d_trend = ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: RSI < 30 (oversold) + above 4h EMA50 + above 1d EMA200 + volume confirmation
            if (rsi_val < 30 and
                price > ema40_trend and
                price > ema1d_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + below 4h EMA50 + below 1d EMA200 + volume confirmation
            elif (rsi_val > 70 and
                  price < ema40_trend and
                  price < ema1d_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or trend turns down
            if (rsi_val > 50 or
                price < ema40_trend or
                price < ema1d_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or trend turns up
            if (rsi_val < 50 or
                price > ema40_trend or
                price > ema1d_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_Trend_Volume"
timeframe = "1h"
leverage = 1.0