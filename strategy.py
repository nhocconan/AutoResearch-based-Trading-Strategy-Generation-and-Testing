#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and 12h EMA trend filter
# Long when price breaks above 4h Donchian upper (20) AND 1d volume > 2.5x 20-period median AND price > 12h EMA50
# Short when price breaks below 4h Donchian lower (20) AND 1d volume > 2.5x 20-period median AND price < 12h EMA50
# Exit when price reverses 2.0x ATR from extreme OR reverts to 12h EMA50
# Uses discrete position size 0.25 to balance capture and fee drag. Target: 75-200 total trades over 4 years.
# Combines price channel breakout (Donchian) with trend filter (12h EMA50) and volume confirmation (1d) for robustness across regimes.

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicators: EMA (50-period) for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    
    # Get ATR for stoploss (using 4h data)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if 'close_4h' in locals() else np.abs(high_4h - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[low_4h[0]], low_4h[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Need close_4h for TR calculation
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50, 14)  # 1d volume, 4h Donchian, 12h EMA, 4h ATR
    
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
            
        # Volume filter: current 1d volume > 2.5x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.5
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
            # Exit when price retraces 2.0x ATR from high OR reverts to EMA50
            if price <= max_favorable_price - 2.0 * atr_aligned[i] or price <= ema50:
                exit_signal = True
        elif position == -1:  # short position
            # Update max favorable price (lowest price for short)
            if price < max_favorable_price or max_favorable_price == 0.0:
                max_favorable_price = price
            # Exit when price rallies 2.0x ATR from low OR reverts to EMA50
            if price >= max_favorable_price + 2.0 * atr_aligned[i] or price >= ema50:
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

name = "4h_Donchian20_1dVolumeSpike_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0