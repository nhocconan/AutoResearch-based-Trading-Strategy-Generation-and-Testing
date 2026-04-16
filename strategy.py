#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly volume confirmation and ATR-based stoploss
# Donchian channels identify key support/resistance levels. Breakouts with volume confirmation
# capture sustained moves in both bull and bear markets. Uses discrete sizing (0.25) to minimize fees.
# Timeframe: 1d, HTF: 1w for volume confirmation. Target: 50-100 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data (higher timeframe for volume confirmation) ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # === 1d Donchian(20) channels ===
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # === 1w volume confirmation (20-period MA) ===
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1w = volume_1w > (1.5 * vol_ma_20_1w)
    
    # Align weekly volume confirmation to daily timeframe
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1w, vol_confirm_1w)
    
    # === 1d ATR(14) for stoploss ===
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        atr_val = atr[i]
        vol_conf = vol_confirm_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches Donchian lower (opposite side)
            if price <= lower:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches Donchian upper (opposite side)
            if price >= upper:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_conf:
                # Go long when price breaks above Donchian upper
                if price > upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian lower
                elif price < lower:
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

name = "1d_Donchian20_Breakout_Volume_ATRStop"
timeframe = "1d"
leverage = 1.0