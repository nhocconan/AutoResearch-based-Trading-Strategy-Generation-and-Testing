#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and ATR-based position sizing
# - Uses 6h Donchian channels (20-period) for breakout entries
# - Requires 1d volume > 1.5 * 20-period volume average for confirmation (to filter weak breakouts)
# - Uses ATR(14) for dynamic stoploss (2.5 * ATR) and scales position size inversely to volatility
# - Works in bull markets via breakouts above upper channel, in bear via breakdowns below lower channel
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to avoid fee drag
# - Donchian channels provide adaptive structure that works in both trending and ranging markets

name = "6h_1d_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 6h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: highest high and lowest low over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h ATR(14) for volatility-based position sizing and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: volume > 1.5 * 20-period average
    df_1d_vol = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_vol).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = df_1d_vol > (1.5 * vol_ma_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < highest_high[i] - 2.5 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif close[i] < midpoint:  # Mean reversion exit
                position = 0
                signals[i] = 0.0
            else:
                # Scale position size inversely to volatility (higher vol = smaller position)
                vol_factor = np.clip(atr[i] / (0.02 * close[i]), 0.5, 2.0)  # Normalize ATR to 2% of price
                base_size = 0.25
                size = base_size / vol_factor
                size = np.clip(size, 0.15, 0.35)  # Keep within reasonable bounds
                signals[i] = size
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > lowest_low[i] + 2.5 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif close[i] > midpoint:  # Mean reversion exit
                position = 0
                signals[i] = 0.0
            else:
                # Scale position size inversely to volatility
                vol_factor = np.clip(atr[i] / (0.02 * close[i]), 0.5, 2.0)
                base_size = 0.25
                size = base_size / vol_factor
                size = np.clip(size, 0.15, 0.35)
                signals[i] = -size
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > highest_high[i] and volume_confirm_aligned[i]:  # Break above upper channel
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and volume_confirm_aligned[i]:  # Break below lower channel
                position = -1
                signals[i] = -0.25
    
    return signals