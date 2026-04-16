#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR-based trend filter
# Uses Donchian(20) breakouts on 4h with volume > 1.5x 20-period average and 
# price above/below 4h EMA34 for trend direction. Exit at opposite Donchian band.
# Works in both bull and bear markets by following price channels with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === Donchian channels (20-period) ===
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === EMA34 for trend filter ===
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Volume confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * vol_ma_20_4h)
    
    # === ATR for stoploss ===
    atr_4h = np.maximum(
        high_4h - low_4h,
        np.maximum(
            np.abs(high_4h - np.roll(close_4h, 1)),
            np.abs(low_4h - np.roll(close_4h, 1))
        )
    )
    atr_4h[0] = high_4h[0] - low_4h[0]  # First value
    atr_ma_4h = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema34_4h[i]) or
            np.isnan(vol_ma_20_4h[i]) or
            np.isnan(atr_ma_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema34 = ema34_4h[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr_ma_4h[i]
        
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
            # Exit when price touches or crosses below lower Donchian band
            if price <= lower_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses above upper Donchian band
            if price >= upper_band:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike
            if vol_spike_val:
                # Go long when price breaks above upper Donchian band and above EMA34
                if price > upper_band and price > ema34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below lower Donchian band and below EMA34
                elif price < lower_band and price < ema34:
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

name = "4h_Donchian_Breakout_Volume_EMA34_Filter"
timeframe = "4h"
leverage = 1.0