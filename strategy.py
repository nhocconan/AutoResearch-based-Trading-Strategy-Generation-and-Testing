#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v3"
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
    
    # Calculate daily pivot for exit
    pivot_1d_val = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3 if len(high_1d) > 0 else 0
    pivot_array = np.full_like(high_1d, pivot_1d_val)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        atr_val = atr[i]
        pivot = pivot_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Camarilla-based signals
        long_signal = False
        short_signal = False
        
        # Long: price breaks above R4 with volume
        if price_high > r4 and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below S4 with volume
        if price_low < s4 and volume_confirmed:
            short_signal = True
        
        # Exit conditions: ATR stop loss or return to daily pivot
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 1.5 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 1.5 * atr_val)
        
        # Exit to pivot
        exit_long = position == 1 and price_close < pivot
        exit_short = position == -1 and price_close > pivot
        
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
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h Camarilla breakout strategy with volume confirmation and ATR stop loss.
# Enters long when price breaks above R4 (strong bullish breakout) with volume confirmation.
# Enters short when price breaks below S4 (strong bearish breakdown) with volume confirmation.
# Exits when price returns to daily pivot point or ATR stop loss is hit.
# Uses volume confirmation (>1.5x 20-period average) to ensure institutional participation.
# Target: 20-40 trades per year to minimize fee decay while capturing strong directional moves.
# Works in both bull and bear markets by trading breakouts in either direction.