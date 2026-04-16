#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above daily Donchian high(20) with volume > 1.5x 20-period average.
# Short when price breaks below daily Donchian low(20) with volume > 1.5x 20-period average.
# Exit when price closes back inside the Donchian channel or ATR stoploss is hit.
# Uses discrete position size 0.25. Daily Donchian provides structure from higher timeframe, 12h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get daily data once before loop for Donchian levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Donchian Channel (20) based on prior day ===
    # Calculate using prior day's high, low, close (shift by 1 to use completed day only)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(phigh).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(plow).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Daily ATR (14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume moving average (20-period) on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop logic (not used in this version)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        dch = donch_high_aligned[i]
        dcl = donch_low_aligned[i]
        dcm = donch_mid_aligned[i]
        atr_val = atr_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price closes back inside Donchian channel (mean reversion)
            if price <= dch and price >= dcl:
                exit_signal = True
            # ATR-based stoploss: exit if price drops 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price closes back inside Donchian channel (mean reversion)
            if price <= dch and price >= dcl:
                exit_signal = True
            # ATR-based stoploss: exit if price rises 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above daily Donchian high with volume confirmation
            if (price > dch) and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below daily Donchian low with volume confirmation
            elif (price < dcl) and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_ATRStop_V1"
timeframe = "12h"
leverage = 1.0