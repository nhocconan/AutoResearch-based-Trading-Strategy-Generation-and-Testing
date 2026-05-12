#!/usr/bin/env python3
name = "6h_RSI50_Cross_ADX20_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d RSI(14) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = np.where(avg_loss == 0, 100, rsi_14_1d)  # Handle no loss case
    
    # Align RSI to 6h
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # === 6h ADX(14) for trend strength ===
    # Calculate +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    up_move = np.insert(up_move, 0, 0)
    down_move = np.insert(down_move, 0, 0)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # First TR
    
    # Smooth with Wilder's method (period=14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[1:period+1]) if len(data) >= period+1 else 0
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
    adx = wilder_smooth(dx, 14)
    
    # === 6h RSI(14) for entry signal ===
    delta_6h = np.diff(close, prepend=close[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(gain_6h)
    avg_loss_6h = np.zeros_like(loss_6h)
    if len(gain_6h) >= 14:
        avg_gain_6h[13] = np.mean(gain_6h[1:14])
        avg_loss_6h[13] = np.mean(loss_6h[1:14])
        for i in range(14, len(gain_6h)):
            avg_gain_6h[i] = (avg_gain_6h[i-1] * 13 + gain_6h[i]) / 14
            avg_loss_6h[i] = (avg_loss_6h[i-1] * 13 + loss_6h[i]) / 14
    
    rs_6h = np.divide(avg_gain_6h, avg_loss_6h, out=np.zeros_like(avg_gain_6h), where=avg_loss_6h!=0)
    rsi_14_6h = 100 - (100 / (1 + rs_6h))
    rsi_14_6h = np.where(avg_loss_6h == 0, 100, rsi_14_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(adx[i]) or
            np.isnan(rsi_14_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Daily RSI > 50 (bullish trend) + ADX > 20 (trending) + 6h RSI crosses above 50
            if (rsi_14_1d_aligned[i] > 50 and
                adx[i] > 20 and
                rsi_14_6h[i] > 50 and
                rsi_14_6h[i-1] <= 50):
                signals[i] = 0.25
                position = 1
            # Short: Daily RSI < 50 (bearish trend) + ADX > 20 (trending) + 6h RSI crosses below 50
            elif (rsi_14_1d_aligned[i] < 50 and
                  adx[i] > 20 and
                  rsi_14_6h[i] < 50 and
                  rsi_14_6h[i-1] >= 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Daily RSI < 50 or 6h RSI < 40
            if rsi_14_1d_aligned[i] < 50 or rsi_14_6h[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Daily RSI > 50 or 6h RSI > 60
            if rsi_14_1d_aligned[i] > 50 or rsi_14_6h[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals