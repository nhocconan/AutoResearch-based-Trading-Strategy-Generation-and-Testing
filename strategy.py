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
    
    # Get 12h data for trend context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(21) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h ADX(14) for trend strength
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.mean(tr[i-14:i+1])
    
    # +DM and -DM
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    up_move = np.insert(up_move, 0, np.nan)
    down_move = np.insert(down_move, 0, np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = np.full(len(df_12h), np.nan)
    minus_dm_smooth = np.full(len(df_12h), np.nan)
    tr_smooth = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        if i == 14:
            plus_dm_smooth[i] = np.sum(plus_dm[1:15])
            minus_dm_smooth[i] = np.sum(minus_dm[1:15])
            tr_smooth[i] = np.sum(tr[1:15])
        else:
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
    
    # DI and DX
    plus_di = np.divide(100 * plus_dm_smooth, tr_smooth, out=np.full_like(plus_dm_smooth, np.nan), where=tr_smooth!=0)
    minus_di = np.divide(100 * minus_dm_smooth, tr_smooth, out=np.full_like(minus_dm_smooth, np.nan), where=tr_smooth!=0)
    dx = np.divide(100 * np.abs(plus_di - minus_di), plus_di + minus_di, out=np.full_like(plus_di, np.nan), where=(plus_di + minus_di)!=0)
    adx_12h = np.full(len(df_12h), np.nan)
    for i in range(27, len(df_12h)):  # 14 + 13 for smoothing
        if not np.isnan(dx[i-13:i+1]):
            adx_12h[i] = np.mean(dx[i-13:i+1])
    
    # Align 12h indicators to 6h timeframe
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr_h[i-14:i+1])
    
    # Calculate 6h volume moving average
    vol_s = pd.Series(volume)
    vol_ma_20_6h = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5 * 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_6h[i]
        
        # Trend filter: price above/below 12h EMA21 with strong trend (ADX > 25)
        above_ema = close[i] > ema_21_12h_aligned[i]
        below_ema = close[i] < ema_21_12h_aligned[i]
        strong_trend = adx_12h_aligned[i] > 25
        
        # Entry conditions: trend + volume + strong trend
        long_entry = above_ema and vol_filter and strong_trend
        short_entry = below_ema and vol_filter and strong_trend
        
        # Exit conditions: trend weakening or reversal
        long_exit = (not above_ema) or (adx_12h_aligned[i] < 20)
        short_exit = (not below_ema) or (adx_12h_aligned[i] < 20)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_ema21_adx_vol_filter_v1"
timeframe = "6h"
leverage = 1.0