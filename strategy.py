# [51406] 4h Donchian Breakout + Volume Spike + Regime Filter
# Hypothesis: Price breakouts from Donchian channels on 4h with volume confirmation and volatility regime filter
# capture true breakouts while avoiding false signals in choppy markets. Works in both bull/bear by
# filtering for high-volume, low-volatility breakouts which indicate institutional participation.
# Target: 20-40 trades/year per symbol with strict entry conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian(20) for entry/exit levels
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = align_htf_to_ltf(prices, df_4h, high_20_4h)
    donchian_lower_4h = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # === 1d data (HTF for regime filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d ATR(14) for volatility regime
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # === Volume indicators ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(atr_1d_ma_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        upper_4h = donchian_upper_4h[i]
        lower_4h = donchian_lower_4h[i]
        vol_ratio_val = vol_ratio[i]
        atr_ma_val = atr_1d_ma_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower
            if price < lower_4h:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper
            if price > upper_4h:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session with volume spike and low volatility regime
            if in_session:
                # Volatility regime: current ATR < 50-day MA ATR (low volatility environment)
                vol_regime = atr_1d[i] < atr_ma_val if not np.isnan(atr_1d[i]) else False
                
                # LONG: Price breaks above Donchian upper with volume spike in low vol regime
                if (price > upper_4h) and (vol_ratio_val > 2.5) and vol_regime:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price breaks below Donchian lower with volume spike in low vol regime
                elif (price < lower_4h) and (vol_ratio_val > 2.5) and vol_regime:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_VolRegime"
timeframe = "4h"
leverage = 1.0