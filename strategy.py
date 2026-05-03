#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper band in 1w uptrend (price > EMA50) with volume > 1.5x MA.
# Short when price breaks below 12h Donchian lower band in 1w downtrend (price < EMA50) with volume > 1.5x MA.
# Uses ATR-based stoploss to limit drawdown. Discrete sizing 0.25 to balance return and risk.
# Donchian channels provide clear breakout levels, 1w EMA50 ensures alignment with major trend,
# Volume confirmation filters false breakouts. Designed for 12h timeframe to minimize fee drag.
# Works in both bull and bear markets by only trading with the 1w trend, avoiding counter-trend whipsaws.

name = "12h_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper and lower bands
    donchian_upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 15m timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (12h ATR 14)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume confirmation (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 1w uptrend AND volume confirmation
            if close_val > donchian_upper_aligned[i] and trend_up and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian lower AND 1w downtrend AND volume confirmation
            elif close_val < donchian_lower_aligned[i] and trend_down and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Check stoploss: 2.5 * ATR below entry
            if close_val < entry_price - 2.5 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Exit long: price breaks below Donchian lower (reversal signal)
            elif close_val < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Check stoploss: 2.5 * ATR above entry
            if close_val > entry_price + 2.5 * atr_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Exit short: price breaks above Donchian upper (reversal signal)
            elif close_val > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals