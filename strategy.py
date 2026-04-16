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
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1w data (HTF for trend context) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # === 1d data (HTF for additional context) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h ATR(15) for volatility and stoploss ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_15_12h = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    atr_15_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_15_12h)
    
    # === 1w EMA30 for long-term trend filter ===
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # === 1d EMA30 for medium-term trend filter ===
    ema_30_1d = pd.Series(close_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    
    # === 12h Donchian(25) for breakout levels ===
    donch_high_12h = pd.Series(high_12h).rolling(window=25, min_periods=25).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=25, min_periods=25).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    # === 12h RSI(14) for momentum filter ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_12h = 100 - (100 / (1 + rs))
    rsi_14_12h[avg_loss == 0] = 100
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(ema_30_1d_aligned[i]) or
            np.isnan(atr_15_12h_aligned[i]) or
            np.isnan(vol_ratio_12h[i]) or
            np.isnan(donch_high_12h_aligned[i]) or
            np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(rsi_14_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend_1w = ema_30_1w_aligned[i]
        ema_trend_1d = ema_30_1d_aligned[i]
        atr = atr_15_12h_aligned[i]
        vol_ratio = vol_ratio_12h[i]
        donch_high = donch_high_12h_aligned[i]
        donch_low = donch_low_12h_aligned[i]
        rsi = rsi_14_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if price < donch_low or price < ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if price > donch_high or price > ema_trend_1d:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian high with volume, in uptrend (above EMA30_1d and EMA30_1w)
            # and RSI not overbought (to avoid chasing pumps)
            if (price > donch_high and vol_ratio > 2.0 and 
                price > ema_trend_1d and price > ema_trend_1w and 
                rsi < 70):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below Donchian low with volume, in downtrend (below EMA30_1d and EMA30_1w)
            # and RSI not oversold (to avoid catching dead cat bounces)
            elif (price < donch_low and vol_ratio > 2.0 and 
                  price < ema_trend_1d and price < ema_trend_1w and 
                  rsi > 30):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1d1wEMA30_Volume_RSI_v1"
timeframe = "12h"
leverage = 1.0