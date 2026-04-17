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
    
    # === 1d ATR (14-period) for volatility ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR with Wilder's smoothing
    atr_1d = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_1d[i] = tr[i]
            else:
                atr_1d[i] = (atr_1d[i-1] * (i-1) + tr[i]) / i
        else:
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # === 1d EMA (50-period) for trend ===
    ema_50 = np.full_like(close_1d, np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50[i] = close_1d[i]
        elif np.isnan(ema_50[i-1]):
            ema_50[i] = close_1d[i]
        else:
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    # === 1d Bollinger Bands (20,2) ===
    sma_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
        elif i > 0:
            sma_20[i] = np.mean(close_1d[max(0, i-9):i+1])
        else:
            sma_20[i] = close_1d[0]
    
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            std_20[i] = np.std(close_1d[i-19:i+1])
        elif i > 0:
            std_20[i] = np.std(close_1d[max(0, i-9):i+1])
        else:
            std_20[i] = 0.0
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # 20-period average volume
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    warmup = 100
    
    # Track position
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: price above/below 50 EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy markets)
        vol_filter = atr_1d_aligned[i] > close[i] * 0.005
        
        if position == 0:
            # Long: uptrend + volatility + price near lower BB + volume confirmation
            if (uptrend and vol_filter and
                close[i] <= lower_bb_aligned[i] * 1.01 and  # within 1% of lower BB
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: downtrend + volatility + price near upper BB + volume confirmation
            elif (downtrend and vol_filter and
                  close[i] >= upper_bb_aligned[i] * 0.99 and  # within 1% of upper BB
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses above middle BB or trend reverses
            if close[i] >= sma_20[-1] if len(sma_20) > 0 else False or not uptrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below middle BB or trend reverses
            if close[i] <= sma_20[-1] if len(sma_20) > 0 else False or not downtrend:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BB_Bounce_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0