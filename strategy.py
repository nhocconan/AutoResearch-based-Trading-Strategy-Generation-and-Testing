#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 12h ATR Regime and Volume Confirmation.
- Elder Ray Bull Power = High - EMA13; Bear Power = EMA13 - Low. Measures bull/bear strength relative to trend.
- 12h ATR(14) regime filter: High volatility (ATR > 20-period mean) favors trend continuation; low volatility favors mean reversion.
- Volume spike (>1.8x 20-period average) confirms institutional participation.
- In high volatility regime: trend follow (long when Bull Power > 0, short when Bear Power > 0).
- In low volatility regime: mean revert (long when Bull Power < 0, short when Bear Power < 0).
- Discrete position sizing (0.25) to manage fee drag on 6h timeframe.
- Target trades: 50-150 total over 4 years (12-37/year) to avoid fee drag.
- Works in bull/bear markets via adaptive regime filter and volume confirmation.
"""

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
    
    # Get 12h data ONCE before loop for ATR regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h ATR(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr_12h = np.maximum(high_12h - low_12h, np.absolute(high_12h - np.roll(close_12h, 1)), np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]  # first period
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20_12h = pd.Series(atr_14_12h).rolling(window=20, min_periods=20).mean().values
    atr_regime_high = atr_14_12h > atr_ma_20_12h  # True = high volatility (trend regime)
    atr_regime_high_aligned = align_htf_to_ltf(prices, df_12h, atr_regime_high)
    
    # Calculate EMA13 for Elder Ray (using 15m equivalent: 12 periods of 6h = 3 days, but we'll use 13 for standard)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_regime_high_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-dependent entry logic
            if atr_regime_high_aligned[i]:  # High volatility = trend following regime
                # Long: Bull Power > 0 (bulls in control) with volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power > 0 (bears in control) with volume spike
                elif bear_power[i] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Low volatility = mean reversion regime
                # Long: Bull Power < 0 (bulls weak) with volume spike (mean reversion long)
                if bull_power[i] < 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (bears weak) with volume spike (mean reversion short)
                elif bear_power[i] < 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: regime change or power deterioration
            if atr_regime_high_aligned[i]:  # Trend regime
                if bull_power[i] <= 0:  # Bulls lose control
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Mean reversion regime
                if bull_power[i] >= 0:  # Bulls recover (mean reversion complete)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: regime change or power deterioration
            if atr_regime_high_aligned[i]:  # Trend regime
                if bear_power[i] <= 0:  # Bears lose control
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Mean reversion regime
                if bear_power[i] >= 0:  # Bears recover (mean reversion complete)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hATRRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0