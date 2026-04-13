#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility and volatility ratio
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Calculate 30-day ATR for volatility regime detection
    atr_30d = np.zeros_like(atr_1d)
    for i in range(len(atr_1d)):
        if i < 30:
            atr_30d[i] = np.mean(atr_1d[:i+1]) if i > 0 else atr_1d[i]
        else:
            atr_30d[i] = 0.97 * atr_30d[i-1] + 0.03 * atr_1d[i]
    
    # Volatility ratio: current ATR / 30-day ATR (regime detection)
    vol_ratio = np.divide(atr_1d, atr_30d, out=np.ones_like(atr_1d), where=atr_30d!=0)
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily RSI for momentum filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[i]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[i]
        else:
            avg_gain[i] = 0.92 * avg_gain[i-1] + 0.08 * gain[i]
            avg_loss[i] = 0.92 * avg_loss[i-1] + 0.08 * loss[i]
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 10-day RSI for short-term momentum
    delta_10 = np.diff(close_1d, prepend=close_1d[0])
    gain_10 = np.where(delta_10 > 0, delta_10, 0)
    loss_10 = np.where(delta_10 < 0, -delta_10, 0)
    avg_gain_10 = np.zeros_like(gain_10)
    avg_loss_10 = np.zeros_like(loss_10)
    for i in range(len(gain_10)):
        if i < 10:
            avg_gain_10[i] = np.mean(gain_10[:i+1]) if i > 0 else gain_10[i]
            avg_loss_10[i] = np.mean(loss_10[:i+1]) if i > 0 else loss_10[i]
        else:
            avg_gain_10[i] = 0.9 * avg_gain_10[i-1] + 0.1 * gain_10[i]
            avg_loss_10[i] = 0.9 * avg_loss_10[i-1] + 0.1 * loss_10[i]
    rs_10 = np.divide(avg_gain_10, avg_loss_10, out=np.zeros_like(avg_gain_10), where=avg_loss_10!=0)
    rsi_10 = 100 - (100 / (1 + rs_10))
    
    # Calculate daily volume moving average for volume filter
    vol_ma_20 = np.zeros_like(vol_1d)
    for i in range(len(vol_1d)):
        if i < 20:
            vol_ma_20[i] = np.mean(vol_1d[:i+1]) if i > 0 else vol_1d[i]
        else:
            vol_ma_20[i] = 0.95 * vol_ma_20[i-1] + 0.05 * vol_1d[i]
    
    # Align indicators to 6h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    rsi_10_aligned = align_htf_to_ltf(prices, df_1d, rsi_10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(rsi_10_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-day average
        volume_filter = volume[i] > vol_ma_20_aligned[i]
        
        # Trend and momentum filters
        uptrend = close[i] > ema_200_1d_aligned[i]
        strong_momentum = rsi_10_aligned[i] > 50
        
        downtrend = close[i] < ema_200_1d_aligned[i]
        weak_momentum = rsi_10_aligned[i] < 50
        
        # Volatility regime filter: only trade in normal/high volatility (avoid low vol chop)
        normal_vol = vol_ratio_aligned[i] >= 0.8
        
        # Entry conditions
        long_entry = uptrend and strong_momentum and volume_filter and normal_vol
        short_entry = downtrend and weak_momentum and volume_filter and normal_vol
        
        # Exit conditions: trend reversal or momentum deterioration
        exit_long = position == 1 and (not uptrend or rsi_10_aligned[i] < 40)
        exit_short = position == -1 and (not downtrend or rsi_10_aligned[i] > 60)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_vol_regime_rsi_ema_filter_v1"
timeframe = "6h"
leverage = 1.0