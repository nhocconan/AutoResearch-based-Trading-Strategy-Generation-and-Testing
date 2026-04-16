#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. Using 1d timeframe reduces noise.
# 1w EMA50 provides long-term trend filter to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have participation.
# ATR-based trailing stop (2.5x ATR) manages risk.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in both bull (breakouts with trend) and bear (short breakdowns against trend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for Donchian calculation (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data for EMA50 trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Donchian Channels (20-period) on 1d ===
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Align Donchian levels to 1d timeframe (no alignment needed as same TF)
    donchian_upper_1d = donchian_upper
    donchian_lower_1d = donchian_lower
    
    # === EMA50 on 1w for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume confirmation (20-period average) on 1d ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = vol_ma_20  # same timeframe
    
    # === ATR calculation (14-period) on 1d for trailing stop ===
    tr1 = pd.Series(high_1d).values - pd.Series(low_1d).values
    tr2 = np.abs(pd.Series(high_1d).values - np.roll(pd.Series(close_1d).values, 1))
    tr3 = np.abs(pd.Series(low_1d).values - np.roll(pd.Series(close_1d).values, 1))
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper_1d[i]) or 
            np.isnan(donchian_lower_1d[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper_1d[i]
        lower = donchian_lower_1d[i]
        ema50 = ema50_1w_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        atr = atr_14[i]
        vol_confirm = volume[i] > vol_ma * 1.5  # 1.5x average volume
        
        # Update trailing stop extreme price
        if position == 1:  # Long position
            if price > extreme_price:
                extreme_price = price
            # Trail stop: exit if price drops 2.5*ATR from extreme
            if atr > 0 and price < extreme_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        elif position == -1:  # Short position
            if price < extreme_price or extreme_price == 0:
                extreme_price = price
            # Trail stop: exit if price rises 2.5*ATR from extreme
            if atr > 0 and price > extreme_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper AND price > 1w EMA50 (uptrend) AND volume confirmation
            if price > upper and price > ema50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: price breaks below Donchian lower AND price < 1w EMA50 (downtrend) AND volume confirmation
            elif price < lower and price < ema50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                extreme_price = price
                continue
        
        # === EXIT LOGIC (Donchian opposite break) ===
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian lower
            if price < lower:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper
            if price > upper:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
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