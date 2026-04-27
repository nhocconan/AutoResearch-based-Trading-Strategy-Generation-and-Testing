#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA(50) on 1d close
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA(20) on 1d volume
    vol_ma_period = 20
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= vol_ma_period:
        for i in range(vol_ma_period-1, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-vol_ma_period+1:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # RSI(14) on 6h close
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rsi = np.full(n, 50.0)
    for i in range(rsi_period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # 6h Donchian(20) breakout
    donch_len = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(donch_len-1, n):
        highest_high[i] = np.max(high[i-donch_len+1:i+1])
        lowest_low[i] = np.min(low[i-donch_len+1:i+1])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    start_idx = max(rsi_period, ema_period, vol_ma_period, donch_len-1)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: RSI > 60, uptrend, volume, price breaks above Donchian high
            if (rsi[i] > 60 and uptrend and volume_confirmation and
                price > highest_high[i]):
                signals[i] = size
                position = 1
            # Short: RSI < 40, downtrend, volume, price breaks below Donchian low
            elif (rsi[i] < 40 and downtrend and volume_confirmation and
                  price < lowest_low[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI < 50 or trend reversal
            if rsi[i] < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI > 50 or trend reversal
            if rsi[i] > 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Momentum_1dTrend_Volume_DonchianBreakout"
timeframe = "6h"
leverage = 1.0