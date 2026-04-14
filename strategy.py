#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d KAMA direction with 12h RSI and volume confirmation.
# 1d KAMA provides adaptive trend direction that avoids whipsaws in ranging markets.
# 12h RSI (14) identifies overbought/oversold conditions for entry in trend direction.
# Volume confirmation (>1.5x 20-period average) filters weak breakouts.
# Exit when RSI returns to neutral (50) or trend reverses.
# Designed for low trade frequency (~15-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d data
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Calculate ER properly
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 10:  # ER period
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Load 12h data ONCE for RSI and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 12h data
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain/loss
    avg_gain = np.zeros_like(close_12h)
    avg_loss = np.zeros_like(close_12h)
    rsi_period = 14
    
    # Initial average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        
        # Subsequent averages
        for i in range(rsi_period+1, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    # Calculate RSI
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average
    vol_12h = df_12h['volume'].values
    vol_ma = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA and RSI warmup
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below KAMA
        bullish_trend = close[i] > kama_aligned[i]
        bearish_trend = close[i] < kama_aligned[i]
        
        if position == 0:
            # Look for RSI extreme entries in trend direction
            # Long: RSI < 30 (oversold) AND bullish trend AND volume
            if (rsi_aligned[i] < 30 and 
                bullish_trend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) AND bearish trend AND volume
            elif (rsi_aligned[i] > 70 and 
                  bearish_trend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend reverses
            if (rsi_aligned[i] >= 50 or 
                close[i] <= kama_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend reverses
            if (rsi_aligned[i] <= 50 or 
                close[i] >= kama_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dKAMA_12hRSI_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0