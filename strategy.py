#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 12h ADX > 25 (trending) AND 4h volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low AND 12h ADX > 25 AND 4h volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, ADX filters for trending markets (works in bull/bear),
# volume confirmation ensures participation. Designed for 4h timeframe with target 80-160 trades over 4 years (20-40/year).
# ATR-based stoploss: exit when price moves against position by 2.5x ATR(14).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    # Handle first value
    atr[0] = tr.iloc[0] if len(tr) > 0 else 0
    
    # Get 12h data once before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: ADX(14) for trend filter ===
    # Calculate True Range
    tr1_12h = pd.Series(high_12h - low_12h)
    tr2_12h = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3_12h = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    # Handle first value
    if len(atr_12h) > 0:
        atr_12h[0] = tr_12h.iloc[0]
    
    # Calculate Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_smooth = pd.Series(atr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di_12h = np.where(atr_12h_smooth != 0, 100 * plus_dm_smooth / atr_12h_smooth, 0)
    minus_di_12h = np.where(atr_12h_smooth != 0, 100 * minus_dm_smooth / atr_12h_smooth, 0)
    
    dx_12h = np.where((plus_di_12h + minus_di_12h) != 0, 
                      100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h), 0)
    adx_12h = pd.Series(dx_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian, 14 for ATR/ADX, 20 for volume MA)
    warmup = 40
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        dh = highest_high[i]
        dl = lowest_low[i]
        vol_spike = volume_spike[i]
        adx = adx_12h_aligned[i]
        atr_val = atr[i]
        
        # === STOPLOSS LOGIC ===
        stop_hit = False
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr_val:
                stop_hit = True
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr_val:
                stop_hit = True
        
        if stop_hit:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === EXIT LOGIC (volatility-based) ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volume spike ends
            if price < dl or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volume spike ends
            if price > dh or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND ADX > 25 (trending) AND volume spike
            if price > dh and adx > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND ADX > 25 (trending) AND volume spike
            elif price < dl and adx > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hADX_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0