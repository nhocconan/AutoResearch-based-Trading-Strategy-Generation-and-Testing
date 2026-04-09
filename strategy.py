#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and ATR-based position sizing
# - Uses 12h Donchian channel (20-period) for breakout entries
# - Requires 1d volume > 1.5 * 20-period volume average for confirmation
# - Uses ATR(14) for dynamic position sizing (inverse volatility: size = 0.30 * (atr_median / atr))
# - Works in bull markets via upside breakouts, in bear via downside breakouts
# - Target: 12-25 trades/year on 12h timeframe (48-100 total over 4 years) to minimize fee drag
# - Donchian channels adapt to volatility, providing robust breakout levels

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bands (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h ATR(14) for volatility-based position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median = np.nanmedian(atr[~np.isnan(atr)]) if np.any(~np.isnan(atr)) else 0.01
    
    # Pre-compute 1d volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_confirm = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate volatility-adjusted position size (0.20 to 0.35 range)
        if atr_median > 0:
            vol_scaling = np.clip(atr_median / atr[i], 0.5, 2.0)  # Limit extreme volatility effects
            base_size = 0.25
            size = base_size * vol_scaling
            size = np.clip(size, 0.20, 0.35)  # Keep within reasonable bounds
        else:
            size = 0.25
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = size
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -size
        else:  # Flat
            # Look for breakout entries with volume confirmation
            if close[i] > donch_high[i] and volume_confirm[i]:  # Break above upper band
                position = 1
                signals[i] = size
            elif close[i] < donch_low[i] and volume_confirm[i]:  # Break below lower band
                position = -1
                signals[i] = -size
    
    return signals