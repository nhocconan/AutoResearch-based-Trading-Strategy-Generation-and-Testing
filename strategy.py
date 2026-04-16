#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike filter and ATR trailing stop
# Long when price breaks above 12h Donchian upper (20) AND 1d volume > 2.0x 20-period median AND price > 12h EMA50
# Short when price breaks below 12h Donchian lower (20) AND 1d volume > 2.0x 20-period median AND price < 12h EMA50
# Exit when price reverses 2.5x ATR from extreme OR reverts to 12h EMA50
# Uses discrete position size 0.25 to balance capture and fee drag. Target: 50-150 total trades over 4 years.
# Combines price channel breakout (Donchian) with trend filter (EMA50) and volume confirmation for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Get 12h data for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian upper/lower
    donch_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    
    # === 12h Indicators: EMA (50-period) for trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get ATR for stoploss (using 12h data)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50, 14)  # 1d volume, 12h Donchian/EMA/ATR
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # track highest price for long, lowest for short
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
                max_favorable_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_median_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2.0x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.0
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Price levels
        price = close[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema50 = ema_50_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Update max favorable price
            if price > max_favorable_price:
                max_favorable_price = price
            # Exit when price retraces 2.5x ATR from high OR reverts to EMA50
            if price <= max_favorable_price - 2.5 * atr_aligned[i] or price <= ema50:
                exit_signal = True
        elif position == -1:  # short position
            # Update max favorable price (lowest price for short)
            if price < max_favorable_price or max_favorable_price == 0.0:
                max_favorable_price = price
            # Exit when price rallies 2.5x ATR from low OR reverts to EMA50
            if price >= max_favorable_price + 2.5 * atr_aligned[i] or price >= ema50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            max_favorable_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Donchian upper AND volume confirmation AND price > EMA50
            if price > upper and vol_confirm and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                max_favorable_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower AND volume confirmation AND price < EMA50
            elif price < lower and vol_confirm and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                max_favorable_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_EMA50_v1"
timeframe = "12h"
leverage = 1.0