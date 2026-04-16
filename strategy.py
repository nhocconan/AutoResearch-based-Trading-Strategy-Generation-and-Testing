#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and ATR-based position sizing.
# Long when price breaks above weekly Donchian(20) high AND volume > 1.5x 20-period average volume.
# Short when price breaks below weekly Donchian(20) low AND volume > 1.5x 20-period average volume.
# Uses ATR(14) for dynamic stoploss (signal→0 when price moves against position by 2x ATR).
# Position size: 0.25 (discrete levels to minimize fee churn).
# Weekly Donchian captures major trend breaks; volume confirmation reduces false breakouts.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(20) channels
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Calculate daily ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian20 + ATR14 + VolMA20 need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        atr = atr_14[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price <= weekly Donchian low OR price moves against position by 2x ATR
            if (price <= donchian_low) or (price < entry_price - 2.0 * atr):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price >= weekly Donchian high OR price moves against position by 2x ATR
            if (price >= donchian_high) or (price > entry_price + 2.0 * atr):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirm = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above weekly Donchian high with volume confirmation
            if (price > donchian_high) and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below weekly Donchian low with volume confirmation
            elif (price < donchian_low) and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "1d"
leverage = 1.0