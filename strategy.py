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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate daily EMA200 for trend filter (1d)
    ema200_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(df_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 198) / 200
    
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(rsi_6h[i]) or
            np.isnan(ema200_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        # RSI thresholds: avoid overbought/oversold extremes
        rsi_overbought = 70
        rsi_oversold = 30
        
        # Calculate dynamic pivot levels based on previous day's range
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        prev_close = close_1d[i-1] if i > 0 else close_1d[0]
        prev_range = prev_high - prev_low
        
        # Camarilla-style pivot levels
        r4 = prev_close + (prev_range * 1.1 / 2)
        s4 = prev_close - (prev_range * 1.1 / 2)
        r3 = prev_close + (prev_range * 1.1 / 4)
        s3 = prev_close - (prev_range * 1.1 / 4)
        
        # Align to 6h timeframe
        r4_6h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r4))[i]
        s4_6h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s4))[i]
        r3_6h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), r3))[i]
        s3_6h = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), s3))[i]
        
        if position == 0:
            # Long: Price breaks above R4 with volume confirmation and RSI not overbought
            if close[i] > r4_6h and volume_ratio > vol_threshold and rsi_6h[i] < rsi_overbought:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S4 with volume confirmation and RSI not oversold
            elif close[i] < s4_6h and volume_ratio > vol_threshold and rsi_6h[i] > rsi_oversold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S3 (more sensitive exit) OR RSI becomes overbought
            if close[i] < s3_6h or rsi_6h[i] > rsi_overbought:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R3 (more sensitive exit) OR RSI becomes oversold
            if close[i] > r3_6h or rsi_6h[i] < rsi_oversold:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_R4S4_Breakout_Volume_RSI_v2"
timeframe = "6h"
leverage = 1.0