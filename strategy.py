# [EXPERIMENT #154286] 4h Donchian Breakout + Volume + Volatility Regime Filter
# Hypothesis: Breakouts confirmed by volume and filtered by low volatility (calm before storm) work in both bull and bear markets.
# Uses 4h timeframe with 1d HTF for trend and volatility filters. Target: 20-50 trades/year to avoid fee drag.
# Entry: Price breaks Donchian(20) high/low + volume > 1.5x average + ATR(14) < SMA(ATR, 50) (low volatility regime)
# Exit: Price crosses back through Donchian middle or ATR expands beyond threshold.
# Position sizing: 0.25 to limit drawdown.

#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 4h ATR for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR for HTF volatility regime (longer-term context)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF volatility filter to 4h
    vol_regime_1d = atr_1d < atr_ma_1d  # Low volatility regime on daily
    vol_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_1d)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_regime_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume filter + low volatility regime
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                vol_regime_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume filter + low volatility regime
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  vol_regime_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR volatility regime breaks
            if close[i] < donchian_mid[i] or not vol_regime_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR volatility regime breaks
            if close[i] > donchian_mid[i] or not vol_regime_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals