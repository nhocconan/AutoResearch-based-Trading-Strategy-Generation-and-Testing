#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter (period=14)
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
    
    # Calculate daily RSI for momentum filter (period=14)
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
    
    # Calculate daily volume moving average (period=20)
    vol_ma_20 = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma_20[i] = np.mean(volume_1d[:i+1]) if i > 0 else volume_1d[i]
        else:
            vol_ma_20[i] = 0.95 * vol_ma_20[i-1] + 0.05 * volume_1d[i]
    
    # Calculate daily ADX for trend strength (period=14)
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and TR
    tr_14 = np.zeros_like(tr_1d)
    plus_dm_14 = np.zeros_like(plus_dm)
    minus_dm_14 = np.zeros_like(minus_dm)
    for i in range(len(tr_1d)):
        if i < 14:
            tr_14[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
            plus_dm_14[i] = np.mean(plus_dm[:i+1]) if i > 0 else plus_dm[i]
            minus_dm_14[i] = np.mean(minus_dm[:i+1]) if i > 0 else minus_dm[i]
        else:
            tr_14[i] = 0.93 * tr_14[i-1] + 0.07 * tr_1d[i]
            plus_dm_14[i] = 0.93 * plus_dm_14[i-1] + 0.07 * plus_dm[i]
            minus_dm_14[i] = 0.93 * minus_dm_14[i-1] + 0.07 * minus_dm[i]
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = np.zeros_like(tr_14)
    for i in range(len(tr_14)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx_1d = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < 14:
            adx_1d[i] = np.mean(dx[:i+1]) if i > 0 else dx[i]
        else:
            adx_1d[i] = 0.93 * adx_1d[i-1] + 0.07 * dx[i]
    
    # Align indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h price range for volatility filter
    price_range = high - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (dead markets)
        vol_filter = price_range[i] > (atr_1d_aligned[i] * 0.5)
        
        # Volume filter: above average volume
        vol_spike = volume[i] > vol_ma_20_aligned[i]
        
        # Trend strength filter: avoid choppy markets
        trend_filter = adx_1d_aligned[i] > 20
        
        # Momentum filter: RSI not extreme
        mom_filter = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        # Entry conditions: long when bullish alignment
        long_entry = vol_filter and vol_spike and trend_filter and mom_filter and (close[i] > close[i-1])
        # Entry conditions: short when bearish alignment
        short_entry = vol_filter and vol_spike and trend_filter and mom_filter and (close[i] < close[i-1])
        
        # Exit conditions: reverse signal or volatility collapse
        exit_long = position == 1 and (not vol_filter or not trend_filter or close[i] < close[i-1])
        exit_short = position == -1 and (not vol_filter or not trend_filter or close[i] > close[i-1])
        
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

name = "12h_1d_rsi_vol_adx_filter_v1"
timeframe = "12h"
leverage = 1.0