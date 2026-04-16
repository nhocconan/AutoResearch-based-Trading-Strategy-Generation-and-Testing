#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with volume confirmation and 4h EMA filter.
# 4h EMA(34) provides trend direction, 1h Donchian(20) provides entry timing.
# Volume > 1.5x average confirms breakout strength.
# Session filter (08-20 UTC) reduces noise trades.
# Position size 0.20 for risk control. Stop loss at 2x ATR.
# Designed to work in bull (breakout continuations) and bear (breakdown continuations).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 4h ATR(14) for stop loss
    tr_4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), np.abs(low_4h - np.roll(close_4h, 1))))
    tr_4h[0] = high_4h[0] - low_4h[0]  # First value
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # === 1h Donchian(20) for entry/exit ===
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h volume ratio for confirmation ===
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / vol_ma_10
    
    # === Session filter (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        ema_34 = ema_34_4h_aligned[i]
        atr_val = atr_14_4h_aligned[i]
        donch_high = donch_high_20[i]
        donch_low = donch_low_20[i]
        vol_ratio_val = vol_ratio[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low or trend changes
            if price < donch_low or price < ema_34:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or trend changes
            if price > donch_high or price > ema_34:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above Donchian high, above 4h EMA, volume confirmation
            if price > donch_high and price > ema_34 and vol_ratio_val > 1.5:
                signals[i] = 0.20
                position = 1
                entry_price = price
                continue
            # Short: price breaks below Donchian low, below 4h EMA, volume confirmation
            elif price < donch_low and price < ema_34 and vol_ratio_val > 1.5:
                signals[i] = -0.20
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_EMA34_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0