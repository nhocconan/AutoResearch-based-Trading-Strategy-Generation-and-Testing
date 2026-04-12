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
    
    # Get 12h data for calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(21) for trend
    close_12h_series = pd.Series(close_12h)
    ema_21_12h = close_12h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h ATR(14) for volatility
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = np.full(len(df_12h), np.nan)
    for i in range(14, len(df_12h)):
        atr_12h[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 12h Donchian(20) channels
    donch_high_12h = np.full(len(high_12h), np.nan)
    donch_low_12h = np.full(len(low_12h), np.nan)
    for i in range(19, len(high_12h)):
        donch_high_12h[i] = np.max(high_12h[i-19:i+1])
        donch_low_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Align indicators to 4h timeframe
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 4h ATR(14) for position sizing
    tr1_h = np.abs(high - low)
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = tr2_h[0] = tr3_h[0] = np.nan
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.mean(tr_h[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR MA(20) to avoid low volatility
        atr_ma_20 = np.full(n, np.nan)
        for j in range(33, n):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_12h_aligned[j-19:j+1])):
                atr_ma_20[j] = np.mean(atr_12h_aligned[j-19:j+1])
        vol_filter = atr_12h_aligned[i] > 0.5 * atr_ma_20[i] if not np.isnan(atr_ma_20[i]) else False
        
        # Trend filter: price relative to 12h EMA21
        uptrend = close[i] > ema_21_12h_aligned[i]
        downtrend = close[i] < ema_21_12h_aligned[i]
        
        # Breakout conditions: price breaks 12h Donchian channels
        breakout_up = close[i] > donch_high_12h_aligned[i]
        breakout_down = close[i] < donch_low_12h_aligned[i]
        
        # Entry conditions: Donchian breakout with trend and volatility filter
        long_entry = breakout_up and vol_filter and uptrend
        short_entry = breakout_down and vol_filter and downtrend
        
        # Exit conditions: price crosses back to 12h Donchian mid-point
        mid_point = (donch_high_12h_aligned[i] + donch_low_12h_aligned[i]) / 2
        long_exit = close[i] < mid_point
        short_exit = close[i] > mid_point
        
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

name = "4h_12h_donchian_ema_trend_filter_v1"
timeframe = "4h"
leverage = 1.0