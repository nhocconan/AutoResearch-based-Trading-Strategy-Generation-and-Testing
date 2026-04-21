#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 4h EMA200
    ema_4h_200 = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d RSI14
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h volume average (20-period)
    vol_1h = prices['volume'].values
    vol_ma_20 = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_4h_200_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        
        if position == 0:
            # Enter long: Price above EMA50 & EMA200, RSI not overbought, volume surge
            if (price_close > ema_4h_50_aligned[i] and
                price_close > ema_4h_200_aligned[i] and
                rsi_1d_aligned[i] < 70 and
                vol_1h[i] > 1.5 * vol_ma_20[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: Price below EMA50 & EMA200, RSI not oversold, volume surge
            elif (price_close < ema_4h_50_aligned[i] and
                  price_close < ema_4h_200_aligned[i] and
                  rsi_1d_aligned[i] > 30 and
                  vol_1h[i] > 1.5 * vol_ma_20[i]):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: Price crosses EMA50 or RSI extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: Price below EMA50 or RSI overbought
                if (price_close < ema_4h_50_aligned[i] or
                    rsi_1d_aligned[i] > 70):
                    exit_signal = True
            elif position == -1:
                # Exit short: Price above EMA50 or RSI oversold
                if (price_close > ema_4h_50_aligned[i] or
                    rsi_1d_aligned[i] < 30):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA50_200_RSI14_Volume1.5x_Session"
timeframe = "1h"
leverage = 1.0