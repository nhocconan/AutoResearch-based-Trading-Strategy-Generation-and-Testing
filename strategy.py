#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume regime filter
# - Donchian(20) from 4h: breakout above upper band = long, below lower band = short
# - 12h volume regime: current 4h volume > 2.0x 12h median volume (avoid low-volume false breakouts)
# - ATR-based trailing stop: exit when price moves against position by 2.0*ATR(14)
# - Designed for 4h timeframe: targets 20-40 trades/year to avoid fee drag
# - Volume regime filter reduces whipsaws in ranging markets
# - Works in bull/bear markets: Donchian breakouts capture momentum, volume filter ensures conviction
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_12h_donchian_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h median volume (for volume regime filter)
    volume_12h = df_12h['volume'].values
    # Calculate rolling median of 12h volume over 24 periods (12 days)
    median_volume_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).median().values
    median_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, median_volume_12h)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper band: highest high over past 20 periods
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low over past 20 periods
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume
    volume_4h = prices['volume'].values
    
    # Pre-compute 4h ATR(14) for trailing stop
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_high = 0.0  # for trailing stop
    lowest_low = 0.0    # for trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_4h[i]) or np.isnan(median_volume_12h_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high for trailing stop
            if close_4h[i] > highest_high:
                highest_high = close_4h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_4h[i] < highest_high - 2.0 * atr_14[i] or close_4h[i] < highest_20[i]:
                position = 0
                highest_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            if close_4h[i] < lowest_low:
                lowest_low = close_4h[i]
            # Exit: trailing stop hit OR price re-enters Donchian channel (failed breakout)
            if close_4h[i] > lowest_low + 2.0 * atr_14[i] or close_4h[i] > lowest_20[i]:
                position = 0
                lowest_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume regime filter
            # Volume regime: current 4h volume > 2.0x 12h median volume
            vol_regime = volume_4h[i] > (2.0 * median_volume_12h_aligned[i])
            
            if vol_regime:
                # Breakout long: price closes above upper Donchian band
                if close_4h[i] > highest_20[i]:
                    position = 1
                    entry_price = close_4h[i]
                    highest_high = close_4h[i]
                    signals[i] = 0.25
                # Breakout short: price closes below lower Donchian band
                elif close_4h[i] < lowest_20[i]:
                    position = -1
                    entry_price = close_4h[i]
                    lowest_low = close_4h[i]
                    signals[i] = -0.25
    
    return signals