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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Calculate 20-period average volume on 12h
    avg_volume_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        avg_volume_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER period = 10, Fast EMA = 2, Slow EMA = 30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder - will fix in loop
    
    # Proper KAMA calculation
    er = np.zeros(len(close_1d))
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-9])
        volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full(len(close_1d), np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get 1d RSI
    rsi = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        gains = np.maximum(np.diff(close_1d[i-13:i+1]), 0)
        losses = -np.minimum(np.diff(close_1d[i-13:i+1]), 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Calculate Chop index for regime detection
    tr = np.zeros(len(close_1d))
    atr = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            np.abs(high_1d[i] - close_1d[i-1]),
            np.abs(low_1d[i] - close_1d[i-1])
        )
    
    # Calculate ATR(14)
    for i in range(14, len(tr)):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Chop index (14-period)
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_sum = np.sum(atr[i-13:i+1])
        true_range = np.max(high_1d[i-13:i+1]) - np.min(low_1d[i-13:i+1])
        if true_range > 0:
            chop[i] = 100 * np.log10(atr_sum / true_range) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_12h_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # KAMA trend condition
        kama_bullish = close[i] > kama_aligned[i]
        kama_bearish = close[i] < kama_aligned[i]
        
        # RSI condition (avoid extremes)
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Chop regime condition (chop > 50 = ranging, chop < 50 = trending)
        # We want to trade in trending markets (chop < 50)
        trending_market = chop_aligned[i] < 50
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and kama_bullish and rsi_not_overbought and trending_market
        short_entry = donchian_breakout_short and vol_confirm and kama_bearish and rsi_not_oversold and trending_market
        
        # Exit conditions: opposite Donchian breakout or KAMA reversal
        exit_long = position == 1 and (donchian_breakout_short or close[i] < kama_aligned[i])
        exit_short = position == -1 and (donchian_breakout_long or close[i] > kama_aligned[i])
        
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

name = "6h_12h_1d_kama_rsi_chop_trend"
timeframe = "6h"
leverage = 1.0