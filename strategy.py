#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI + Weekly Trend Filter for 1d timeframe.
# Use weekly EMA(34) to determine trend: price > EMA34 = bullish, price < EMA34 = bearish.
# In bullish trend: long when RSI(14) crosses above 30, short when RSI crosses below 70.
# In bearish trend: short when RSI crosses below 70, long when RSI crosses above 30.
# Volume confirmation: volume > 1.2x 20-period average.
# Target: 15-25 trades/year per symbol to stay within frequency limits.
name = "1d_RSI_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d timeframe (already aligned, but using for consistency)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 1d average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and RSI are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.2 * vol_ma
        
        # Trend determination
        is_bullish = price > ema34
        is_bearish = price < ema34
        
        if position == 0:
            # Determine entry based on trend
            if is_bullish and volume_confirmed:
                # Bullish trend: long on RSI > 30, short on RSI < 70
                if rsi_val > 30 and (i == start_idx or rsi_aligned[i-1] <= 30):
                    signals[i] = 0.25
                    position = 1
                elif rsi_val < 70 and (i == start_idx or rsi_aligned[i-1] >= 70):
                    signals[i] = -0.25
                    position = -1
            elif is_bearish and volume_confirmed:
                # Bearish trend: short on RSI < 70, long on RSI > 30
                if rsi_val < 70 and (i == start_idx or rsi_aligned[i-1] >= 70):
                    signals[i] = -0.25
                    position = -1
                elif rsi_val > 30 and (i == start_idx or rsi_aligned[i-1] <= 30):
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Long exit: RSI crosses below 50 or opposite extreme
            if rsi_val < 50 and (i == start_idx or rsi_aligned[i-1] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50 or opposite extreme
            if rsi_val > 50 and (i == start_idx or rsi_aligned[i-1] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals