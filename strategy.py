#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and ATR-based position sizing.
    # Camarilla levels act as intraday support/resistance with statistical significance.
    # Breakout above H3 or below L3 with volume confirmation captures institutional moves.
    # Works in both bull and bear markets by trading breakouts in direction of trend.
    # Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
    # Target: 50-150 total trades over 4 years (12-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility scaling and stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range components
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # Seed with simple average
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # H3 = Pivot + 1.1 * (H - L) / 2
    # L3 = Pivot - 1.1 * (H - L) / 2
    # H4 = Pivot + 1.1 * (H - L)
    # L4 = Pivot - 1.1 * (H - L)
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_h3 = pivot + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_l3 = pivot - 1.1 * (prev_high - prev_low) / 2.0
    camarilla_h4 = pivot + 1.1 * (prev_high - prev_low)
    camarilla_l4 = pivot - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 12h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period MA
        volume_filter = volume[i] > 1.3 * volume_ma[i]
        
        # Breakout conditions: price breaks above H3 or below L3
        long_breakout = close[i] > h3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Dynamic position size based on volatility (ATR/price ratio)
        # Normalize ATR ratio to target size 0.25, max 0.30
        atr_ratio = atr_1d_aligned[i] / close[i]
        base_size = 0.25
        vol_scaled_size = base_size * (atr_ratio / 0.015)  # Normalize to 1.5% ATR/price
        position_size = np.clip(vol_scaled_size, 0.0, 0.30)
        
        # Entry conditions: breakout with volume confirmation
        long_entry = long_breakout and volume_filter
        short_entry = short_breakout and volume_filter
        
        # Exit conditions: opposite breakout or price reaches H4/L4
        long_exit = short_breakout or (close[i] >= h4_aligned[i])
        short_exit = long_breakout or (close[i] <= l4_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_atr_size_v1"
timeframe = "12h"
leverage = 1.0