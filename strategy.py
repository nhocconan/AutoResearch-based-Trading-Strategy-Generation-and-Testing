#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation + ATR trailing stop
# Donchian(20) provides clear breakout levels with proven edge on SOLUSDT.
# 12h EMA50 acts as trend filter: only long when price > EMA50, short when price < EMA50.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have participation.
# ATR trailing stop (2.5x ATR) manages risk and adapts to volatility.
# This combination worked in experiment #52133 (Sharpe=0.472) and can be optimized for BTC/ETH.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data for EMA50 trend filter (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === EMA50 on 12h ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume confirmation on 12h (20-period average) ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # === Donchian(20) on primary 4h timeframe ===
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # === ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema50 = ema50_aligned[i]
        vol_ma = vol_ma_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_confirm = volume[i] > vol_ma * 1.5  # 1.5x average volume
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Trail stop: exit if price drops 2.5*ATR from extreme
            if atr_val > 0 and price < extreme_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update extreme price (lowest since entry)
            if price < extreme_price or extreme_price == 0:
                extreme_price = price
            # Trail stop: exit if price rises 2.5*ATR from extreme
            if atr_val > 0 and price > extreme_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === EXIT LOGIC (Donchian break of opposite side) ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND price > EMA50 AND volume confirmation
            if price > upper and price > ema50 and vol_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = price
                extreme_price = price
                continue
            # Short when: price breaks below Donchian low AND price < EMA50 AND volume confirmation
            elif price < lower and price < ema50 and vol_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = price
                extreme_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0