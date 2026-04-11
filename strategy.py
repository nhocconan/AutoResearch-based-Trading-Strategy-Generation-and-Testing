#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h close for Camarilla levels
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous 12h close, high, low (shifted by 1 for completed bar)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    # Calculate Camarilla levels for 12h
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_high = prev_close_12h + 1.5 * (prev_high_12h - prev_low_12h)
    camarilla_low = prev_close_12h - 1.5 * (prev_high_12h - prev_low_12h)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Calculate 4h 20-period EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA20 warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Long conditions: price breaks above camarilla_high AND price > EMA20 with volume
        long_signal = volume_confirmed and (price_close > camarilla_high_aligned[i]) and (price_close > ema_20[i])
        
        # Short conditions: price breaks below camarilla_low AND price < EMA20 with volume
        short_signal = volume_confirmed and (price_close < camarilla_low_aligned[i]) and (price_close < ema_20[i])
        
        # Exit when price returns to EMA20 (mean reversion)
        exit_long = position == 1 and (price_close < ema_20[i])
        exit_short = position == -1 and (price_close > ema_20[i])
        
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

# Hypothesis: Camarilla breakout on 12h with volume confirmation and EMA20 trend filter on 4h.
# Uses 12h Camarilla levels (H4/L4) as key support/resistance from higher timeframe.
# Enters long when price breaks above H4 with volume confirmation and above 4h EMA20.
# Enters short when price breaks below L4 with volume confirmation and below 4h EMA20.
# Exits when price returns to 4h EMA20, capturing mean reversion after breakout.
# Works in both bull and bear markets by using 12h structure and 4h trend filter.
# Target: 20-50 trades per year to minimize fee drag on 4h timeframe.