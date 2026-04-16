#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20-day) with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.5x 20-day average
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.5x 20-day average
# ATR trailing stop (2.5x ATR) to manage risk
# Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag on 1d timeframe
# Donchian breakouts capture strong trends, EMA50 filter ensures trend alignment, volume adds conviction
# 1d timeframe avoids overtrading while capturing major moves in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w EMA50 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for spike
        atr_val = atr[i]
        
        # Donchian breakout signals
        breakout_up = price > donchian_upper
        breakout_down = price < donchian_lower
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: breakout above upper band AND price > 1w EMA50 AND volume spike
            if breakout_up and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: breakout below lower band AND price < 1w EMA50 AND volume spike
            elif breakout_down and price < ema_50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_ATRTrail"
timeframe = "1d"
leverage = 1.0