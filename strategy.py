#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation.
# Uses weekly EMA for trend direction and daily Donchian(20) breakouts for entries.
# Volume > 1.5x average confirms breakout strength.
# Designed to capture trends in both bull and bear markets with low trade frequency.
# Target: 7-25 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    # === 1w data (higher timeframe for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily Donchian(20) channels ===
    # Use previous day's high/low to avoid look-ahead
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    high_shift[0] = high[0]  # first bar uses its own value
    low_shift[0] = low[0]
    
    # Calculate rolling max/min of previous day's high/low
    donch_high = pd.Series(high_shift).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_shift).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_20_1w_aligned[i]
        vol = vol_ratio[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Exit if price closes below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit if price closes above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol > 1.5:
                # Long when price breaks above Donchian high AND above weekly EMA
                if price > upper and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short when price breaks below Donchian low AND below weekly EMA
                elif price < lower and price < ema_trend:
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

name = "1d_Donchian20_20EMA1w_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0