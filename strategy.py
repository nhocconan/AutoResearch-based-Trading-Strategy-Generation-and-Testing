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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data for additional context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly ATR (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    low_close_1w = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr_1w[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Weekly RSI (14-period)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.full(len(df_1w), np.nan)
    avg_loss_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        avg_gain_1w[13] = np.mean(gain_1w[:14])
        avg_loss_1w[13] = np.mean(loss_1w[:14])
        for i in range(14, len(df_1w)):
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.full_like(avg_gain_1w, np.nan), where=avg_loss_1w!=0)
    rsi_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if not np.isnan(rs_1w[i]):
            rsi_1w[i] = 100 - (100 / (1 + rs_1w[i]))
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close_1d = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Weekly 6-period EMA for trend
    ema_6_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 6:
        ema_6_1w[5] = np.mean(close_1w[:6])
        multiplier = 2 / (6 + 1)
        for i in range(6, len(df_1w)):
            ema_6_1w[i] = (close_1w[i] * multiplier) + (ema_6_1w[i-1] * (1 - multiplier))
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        vol_ma_1d[19] = np.mean(vol_1d[:20])
        for i in range(20, len(df_1d)):
            vol_ma_1d[i] = (vol_ma_1d[i-1] * 19 + vol_1d[i]) / 20
    
    # Align weekly indicators to 6h timeframe
    atr_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    rsi_6h = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_6_6h = align_htf_to_ltf(prices, df_1w, ema_6_1w)
    
    # Align daily indicators to 6h timeframe
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(rsi_6h[i]) or
            np.isnan(ema_6_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(atr_1d_6h[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 1.0% of price (avoid low volatility periods)
        if atr_1d_6h[i] / close[i] < 0.01:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-day average
        if vol_ma_6h[i] > 0 and volume[i] < vol_ma_6h[i] * 1.3:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly RSI < 40 (not oversold) AND price > weekly EMA6 AND break above 6h Donchian high
            if rsi_6h[i] < 40 and close[i] > ema_6_6h[i] and close[i] > donch_high[i]:
                position = 1
                signals[i] = position_size
            # Short: Weekly RSI > 60 (not overbought) AND price < weekly EMA6 AND break below 6h Donchian low
            elif rsi_6h[i] > 60 and close[i] < ema_6_6h[i] and close[i] < donch_low[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR weekly RSI > 50
            if close[i] < donch_low[i] or rsi_6h[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR weekly RSI < 50
            if close[i] > donch_high[i] or rsi_6h[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_RSI_EMA_Donchian_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0