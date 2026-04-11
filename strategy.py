#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (standard multipliers)
    r4_1d = close_1d + range_1d * 1.1 / 2
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align daily pivots to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Filter: Avoid sideways markets - require price to be outside 15% of daily range
    price_position = (close - s4_1d_aligned) / (r4_1d_aligned - s4_1d_aligned + 1e-10)
    in_extreme_zone = (price_position < 0) | (price_position > 1)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(in_extreme_zone[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        atr_val = atr[i]
        extreme_zone = in_extreme_zone[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals - only in extreme zones to avoid whipsaws
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R4 with volume and in extreme zone
        if price_high > r4 and volume_confirmed and extreme_zone:
            long_signal = True
        
        # Short: price breaks below S4 with volume and in extreme zone
        if price_low < s4 and volume_confirmed and extreme_zone:
            short_signal = True
        
        # Exit conditions
        # Calculate daily pivot for exit
        if len(high_1d) > 0:
            pivot_1d_val = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3
        else:
            pivot_1d_val = 0
        pivot_array = np.full_like(high_1d, pivot_1d_val)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)[i]
        
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        
        # Exit to pivot
        exit_long = position == 1 and price_close < pivot_aligned
        exit_short = position == -1 and price_close > pivot_aligned
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Camarilla breakout strategy with volume confirmation, extreme zone filter, and ATR stop loss.
# Enters long when price breaks above R4 with volume confirmation and only in extreme zones (<0 or >1).
# Enters short when price breaks below S4 with volume confirmation and only in extreme zones.
# Uses volume confirmation (>1.5x 20-period average) to ensure institutional participation.
# Extreme zone filter prevents whipsaws in sideways markets by only allowing breaks outside daily range.
# Exits when price returns to daily pivot point or ATR stop loss (2x) is hit.
# Target: 20-40 trades per year to minimize fee decay while capturing strong directional moves.
# Works in both bull and bear markets by trading breakouts in either direction.