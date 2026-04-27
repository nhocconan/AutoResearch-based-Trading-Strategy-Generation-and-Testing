#!/usr/bin/env python3
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
    
    # Get 1d data for multiple indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 34-day EMA for trend
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Calculate 14-day ATR
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-day standard deviation for volatility
    stddev_20 = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        stddev_20[i] = np.std(close_1d[i-19:i+1])
    
    # Calculate volume moving average
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    stddev_20_aligned = align_htf_to_ltf(prices, df_1d, stddev_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4-hour Bollinger Bands (20, 2)
    bb_period = 20
    bb_mid = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    
    if n >= bb_period:
        for i in range(bb_period-1, n):
            bb_mid[i] = np.mean(close[i-bb_period+1:i+1])
            bb_std[i] = np.std(close[i-bb_period+1:i+1])
    
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(34, bb_period-1, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(stddev_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(bb_mid[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        ema_trend = ema34_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        bb_low = bb_lower[i]
        bb_high = bb_upper[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = stddev_20_aligned[i] > 0.5 * np.mean(stddev_20_aligned[max(0, i-50):i+1])
        
        # Volume confirmation: current volume above average
        vol_confirm = vol > vol_ma
        
        if position == 0:
            # Long: Oversold RSI + price at/below lower BB + uptrend + vol confirm
            if (rsi < 30 and price <= bb_low and price > ema_trend and vol_confirm and vol_filter):
                signals[i] = size
                position = 1
            # Short: Overbought RSI + price at/above upper BB + downtrend + vol confirm
            elif (rsi > 70 and price >= bb_high and price < ema_trend and vol_confirm and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or price reaches middle band
            if rsi > 50 or price >= bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral or price reaches middle band
            if rsi < 50 or price <= bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Bollinger_RSI_EMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0