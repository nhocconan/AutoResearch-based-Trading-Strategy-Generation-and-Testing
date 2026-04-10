#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# - Long: Price breaks above Donchian upper channel (20-period high) + 1d ATR(14) > 1.5x 50-period MA (high volatility regime) + current volume > 20-period MA
# - Short: Price breaks below Donchian lower channel (20-period low) + 1d ATR(14) > 1.5x 50-period MA + current volume > 20-period MA
# - Exit: Price returns to Donchian midpoint (mean reversion) OR ATR regime ends (ATR < 1.2x 50-period MA)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Donchian breakouts capture strong moves, ATR filter ensures volatility expansion confirms breakout validity,
#   volume confirms institutional participation. Targets ~12-37 trades/year on 12h timeframe.

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian Channel(20) for 12h timeframe
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    dc_upper = highest_high
    dc_lower = lowest_low
    dc_mid = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1d ATR(14) and its 50-period MA for regime filter
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_50_1d = pd.Series(atr_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate volume confirmation: current volume > 20-period MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_mid[i]) or 
            np.isnan(atr_ma_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period MA
        vol_confirm = volume[i] > volume_ma_20[i]
        
        # ATR regime filter: 1d ATR > 1.5x 50-period MA (high volatility regime)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, atr_1d)
        atr_regime = vol_1d_current[i] > 1.5 * atr_ma_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for Donchian breakouts
            # Long entry: Price breaks above Donchian upper + vol confirmation + ATR regime
            if close[i] > dc_upper[i] and vol_confirm and atr_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + vol confirmation + ATR regime
            elif close[i] < dc_lower[i] and vol_confirm and atr_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint OR ATR regime ends
            if position == 1:  # Long position
                if close[i] <= dc_mid[i] or not atr_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= dc_mid[i] or not atr_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals