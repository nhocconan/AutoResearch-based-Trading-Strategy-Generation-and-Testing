#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ATR(14) > 0.8x its 50-period EMA (high volatility regime) AND volume > 1.3x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ATR(14) > 0.8x its 50-period EMA (high volatility regime) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Donchian captures breakouts, 1d ATR/EMA filter ensures trades only in high volatility regimes (avoids chop),
# volume spike confirms participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets by trading expansion phases.
# Target: 75-150 trades over 4 years (19-38/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR/EMA calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ATR(14) and its EMA(50) for volatility regime filter ===
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # EMA of ATR
    atr_ema_50_1d = pd.Series(atr_14_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volatility regime: ATR > 0.8 * ATR_EMA (high volatility)
    vol_regime = atr_14_1d > (0.8 * atr_ema_50_1d)
    
    # Align 1d indicators to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume MA, 14 for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_regime_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_regime = vol_regime_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Donchian(20) low or volume spike ends or volatility regime ends
            if price < lower_channel or not vol_spike or not vol_regime:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Donchian(20) high or volume spike ends or volatility regime ends
            if price > upper_channel or not vol_spike or not vol_regime:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND high volatility regime AND volume spike
            if price > upper_channel and vol_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian(20) low AND high volatility regime AND volume spike
            elif price < lower_channel and vol_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolRegime_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0