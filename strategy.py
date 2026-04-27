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
    
    # Get 4h data for trend and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_4h = np.full(len(df_4h), np.nan)
    donchian_low_4h = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-19:i+1])
        donchian_low_4h[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h volume SMA (20-period)
    volume_sma_4h = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        volume_sma_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = np.full(len(df_4h), np.nan)
    alpha_50 = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i < 49:
            ema_50_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_50_4h[i-1]):
                ema_50_4h[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_50_4h[i] = close_4h[i] * alpha_50 + ema_50_4h[i-1] * (1 - alpha_50)
    
    # Align 4h indicators to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    volume_sma_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 14-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(19, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(volume_sma_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(rsi_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility regime filter: ATR > 1.5 * mean ATR (volatility expansion)
        vol_expansion = atr_14_1d_aligned[i] > 1.5 * np.nanmean(atr_14_1d_aligned[max(0, i-48):i+1])
        
        # Volume confirmation: current volume > 1.5 * 4h average volume
        vol_confirm = vol > 1.5 * volume_sma_4h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high + volume confirmation + volatility expansion + uptrend
            if (price > donchian_high_4h_aligned[i] and 
                vol_confirm and 
                vol_expansion and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low + volume confirmation + volatility expansion + downtrend
            elif (price < donchian_low_4h_aligned[i] and 
                  vol_confirm and 
                  vol_expansion and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below 4h Donchian low or trend turns down
            if (price < donchian_low_4h_aligned[i] or 
                ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price breaks above 4h Donchian high or trend turns up
            if (price > donchian_high_4h_aligned[i] or 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_VolumeVol_4hTrend_v1"
timeframe = "1h"
leverage = 1.0