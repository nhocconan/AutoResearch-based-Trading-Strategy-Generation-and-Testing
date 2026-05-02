#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses 4h primary timeframe for low trade frequency (target: 25-35 trades/year)
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# 1d ATR regime filter: only trade when ATR(14) < ATR(50) * 1.2 (low volatility regimes)
# Volume confirmation (>1.5 * 20-period EMA) ensures institutional participation
# Fixed position size 0.25 to balance risk and reward
# Designed for minimal trades: ~30 trades/year per symbol with clear entry/exit rules
# Works in bull markets via breakout continuation and bear markets via volatility contraction breakouts

name = "4h_Donchian20_1dATRRegime_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr_14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr1).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR regime: low volatility when ATR(14) < ATR(50) * 1.2
    atr_regime = atr_14 < (atr_50 * 1.2)
    
    # Align ATR regime to 4h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (4h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(atr_regime_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade in low volatility regimes
            if atr_regime_aligned[i]:
                # Long: price breaks above Donchian high with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low with volume spike
                elif close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop in high volatility regimes
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals