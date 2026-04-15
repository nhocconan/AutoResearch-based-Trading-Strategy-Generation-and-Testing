#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR volatility filter.
# Uses 1d Donchian(20) for breakout signals, filtered by 12h volume spike and ATR-based volatility regime.
# Designed for very low trade frequency (<15/year) to minimize fee drag on 12h timeframe.
# Works in bull/bear: Donchian breakouts capture sustained momentum, volume filter avoids false breakouts,
# ATR filter adapts to changing volatility regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Donchian upper = max(high, lookback=20)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, lookback=20)
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # === 12h Indicators: ATR(14) for volatility regime ===
    # True Range calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise on 12h)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility filter: ATR > 0.5 * 50-period ATR SMA (avoid low volatility choppy periods)
        atr_sma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        vol_regime = atr_14[i] > (atr_sma_50[i] * 0.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_sma_50[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Donchian upper (20-period high)
        # 2. Volume confirmation
        # 3. Volatility regime filter (avoid extremely low volatility)
        if (close[i] > donch_high_aligned[i] and
            vol_confirm and
            vol_regime):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Donchian lower (20-period low)
        # 2. Volume confirmation
        # 3. Volatility regime filter
        elif (close[i] < donch_low_aligned[i] and
              vol_confirm and
              vol_regime):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_VolATR_Filter_v1"
timeframe = "12h"
leverage = 1.0