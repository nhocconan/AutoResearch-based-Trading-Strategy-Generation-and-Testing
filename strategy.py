#1/1
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from math import log

# Hypothesis: 6h Relative Strength Index (RSI) with weekly moving average filter and volume confirmation.
# RSI captures overbought/oversold conditions, while weekly MA defines the primary trend.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets
# by taking long signals only when price > weekly MA (uptrend) and short when price < weekly MA (downtrend).
# Target: 12-37 trades per year (50-150 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Calculate weekly high and low for RSI calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly EMA20 to 6h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate RSI(14) on weekly timeframe
    # First calculate price changes
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = np.zeros(len(close_1w))
    avg_loss = np.zeros(len(close_1w))
    avg_gain[0] = np.mean(gain[:14]) if len(gain) >= 14 else 0
    avg_loss[0] = np.mean(loss[:14]) if len(loss) >= 14 else 0
    
    for i in range(1, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(avg_loss == 0, 100, rsi_1w)
    rsi_1w = np.where(avg_gain == 0, 0, rsi_1w)
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate average volume (24-period = 6 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_ema = ema20_1w_aligned[i]
        rsi_value = rsi_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: RSI < 30 (oversold) + above weekly EMA20 + volume confirmation
            if (rsi_value < 30 and
                price > weekly_ema and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + below weekly EMA20 + volume confirmation
            elif (rsi_value > 70 and
                  price < weekly_ema and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or trend turns down
            if (rsi_value > 50 or
                price < weekly_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 or trend turns up
            if (rsi_value < 50 or
                price > weekly_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_RSI_Trend_Volume"
timeframe = "6h"
leverage = 1.0