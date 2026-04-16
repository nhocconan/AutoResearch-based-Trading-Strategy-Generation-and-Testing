#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and 4h volume spike confirmation.
# Long when price breaks above Donchian upper (20) AND price > 1d EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower (20) AND price < 1d EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses Donchian midline (10-period average of high/low) OR volume drops below average.
# Uses discrete position size 0.25. Donchian captures structure, EMA50 filters higher timeframe trend, volume confirms breakout strength.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = donchian_channel(high, low, 20)
    
    # === 4h Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma_20[i]
        ema50_val = ema50_1d_aligned[i]
        
        # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian middle OR volume drops below average
            if price < donch_middle[i] or vol < vol_ma_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian middle OR volume drops below average
            if price > donch_middle[i] or vol < vol_ma_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND price > 1d EMA50 AND volume spike
            if price > donch_upper[i] and price > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND price < 1d EMA50 AND volume spike
            elif price < donch_lower[i] and price < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_Strict_V1"
timeframe = "4h"
leverage = 1.0