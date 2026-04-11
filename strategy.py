#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return signals
    
    # Calculate 12h close for Camarilla levels
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_h4 = np.zeros_like(close_12h)
    camarilla_l3 = np.zeros_like(close_12h)
    camarilla_h3 = np.zeros_like(close_12h)
    camarilla_l4 = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        if i < 1:
            camarilla_h4[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Use previous bar's high, low, close (already closed)
            ph = high_12h[i-1]
            pl = low_12h[i-1]
            pc = close_12h[i-1]
            camarilla_h4[i] = pc + 1.1 * (ph - pl) / 2
            camarilla_l3[i] = pc - 1.1 * (ph - pl) / 6
            camarilla_h3[i] = pc + 1.1 * (ph - pl) / 4
            camarilla_l4[i] = pc - 1.1 * (ph - pl) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        h4 = camarilla_h4_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l4 = camarilla_l4_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Entry signals - Camarilla level breaks with volume
        long_signal = False
        short_signal = False
        
        # Long: price breaks above H4 with volume
        if price_high > h4 and volume_confirmed:
            long_signal = True
        
        # Short: price breaks below L3 with volume
        if price_low < l3 and volume_confirmed:
            short_signal = True
        
        # Exit conditions
        # Exit long if price drops below H3
        exit_long = position == 1 and price_close < h3
        # Exit short if price rises above L4
        exit_short = position == -1 and price_close > l4
        
        # Stop loss conditions (2x ATR)
        stop_long = position == 1 and price_low < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.0 * atr_val)
        
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

# Hypothesis: 4h Camarilla breakout strategy with volume confirmation.
# Enters long when price breaks above 12h Camarilla H4 level with volume confirmation (>1.5x avg volume).
# Enters short when price breaks below 12h Camarilla L3 level with volume confirmation.
# Uses Camarilla levels from higher timeframe (12h) for institutional-grade support/resistance.
# Volume filter ensures institutional participation and reduces false breakouts.
# Exits when price returns to opposite Camarilla level (H3 for longs, L4 for shorts) or ATR stop loss (2x) is hit.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by trading institutional level breaks in either direction.