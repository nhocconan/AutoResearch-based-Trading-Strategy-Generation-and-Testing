#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels; 1d EMA34 filters for higher timeframe trend direction
# Volume confirmation ensures breakouts have participation. ATR-based stoploss manages risk.
# Target: 80-160 total trades over 4 years (20-40/year) for optimal balance of opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA34 trend filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Donchian(20) channels ===
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR(14) for stoploss ===
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema34 = ema34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma * 1.5
        
        # === TRAILING STOP LOGIC (ATR-based) ===
        if position == 1:  # Long position
            # Trail stop: exit if price drops 2.5*ATR from highest since entry
            if not hasattr(generate_signals, 'long_extreme'):
                generate_signals.long_extreme = price
            else:
                if price > generate_signals.long_extreme:
                    generate_signals.long_extreme = price
                if atr_val > 0 and price < generate_signals.long_extreme - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    delattr(generate_signals, 'long_extreme')
                    continue
        
        elif position == -1:  # Short position
            # Trail stop: exit if price rises 2.5*ATR from lowest since entry
            if not hasattr(generate_signals, 'short_extreme'):
                generate_signals.short_extreme = price
            else:
                if price < generate_signals.short_extreme:
                    generate_signals.short_extreme = price
                if atr_val > 0 and price > generate_signals.short_extreme + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    delattr(generate_signals, 'short_extreme')
                    continue
        
        # === EXIT LOGIC (Donchian opposite break) ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'long_extreme'):
                    delattr(generate_signals, 'long_extreme')
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
                if hasattr(generate_signals, 'short_extreme'):
                    delattr(generate_signals, 'short_extreme')
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND above 1d EMA34 AND volume confirmation
            if price > upper and price > ema34 and vol_confirm:
                signals[i] = 0.25
                position = 1
                generate_signals.long_extreme = price
                continue
            # Short when: price breaks below Donchian low AND below 1d EMA34 AND volume confirmation
            elif price < lower and price < ema34 and vol_confirm:
                signals[i] = -0.25
                position = -1
                generate_signals.short_extreme = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    # Clean up
    if hasattr(generate_signals, 'long_extreme'):
        delattr(generate_signals, 'long_extreme')
    if hasattr(generate_signals, 'short_extreme'):
        delattr(generate_signals, 'short_extreme')
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0