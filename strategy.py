#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return signals
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous 12h bar's high, low, close for today's pivot
    ph_12h = np.roll(high_12h, 1)
    pl_12h = np.roll(low_12h, 1)
    pc_12h = np.roll(close_12h, 1)
    ph_12h[0] = pl_12h[0] = pc_12h[0] = np.nan
    
    # Camarilla levels: H3, L3
    camarilla_h3 = pc_12h + (ph_12h - pl_12h) * 1.1 / 6
    camarilla_l3 = pc_12h - (ph_12h - pl_12h) * 1.1 / 6
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after EMA50 and volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long conditions: price breaks above H3 AND above 1d EMA50 with volume
        long_signal = volume_confirmed and (price_high > camarilla_h3_aligned[i]) and (price_close > ema_50_1d_aligned[i])
        
        # Short conditions: price breaks below L3 AND below 1d EMA50 with volume
        short_signal = volume_confirmed and (price_low < camarilla_l3_aligned[i]) and (price_close < ema_50_1d_aligned[i])
        
        # Exit when price returns to the opposite Camarilla level
        exit_long = position == 1 and price_low < camarilla_l3_aligned[i]
        exit_short = position == -1 and price_high > camarilla_h3_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout on 12h with 1d EMA50 trend filter and volume confirmation on 4h.
# Uses 12h Camarilla levels (H3/L3) as breakout triggers. Enters long when price breaks above H3
# with volume confirmation (>1.5x average) and price above 1d EMA50 (uptrend). Enters short when
# price breaks below L3 with volume confirmation and price below 1d EMA50 (downtrend).
# Exits when price returns to the opposite Camarilla level (L3 for longs, H3 for shorts).
# Camarilla levels are derived from previous day's range and provide institutional support/resistance.
# The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures participation from market actors. Target: 20-50 trades/year to minimize
# fee drag on 4h timeframe. Works in both bull and bear markets by aligning with the higher timeframe trend.