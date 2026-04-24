#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d ATR regime filter + volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR-based regime filter.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price.
  Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
- Regime filter: 1d ATR(14) > 1.5 * 50-period ATR MA = high volatility (trade Alligator signals).
- Volume confirmation: current volume > 1.3 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying bullish Alligator alignment, in bear via selling bearish alignment,
  and avoids low-volatility chop where Alligator whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR and its MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - Wilder's smoothing
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 50-period MA of ATR(14) for regime threshold
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Regime: high volatility when ATR(14) > 1.5 * 50-period ATR MA
    high_vol_regime = atr_14 > (1.5 * atr_ma_50)
    
    # Align 1d regime to 12h timeframe (completed 1d bar only)
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    # Williams Alligator on 12h
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # ATR MA(50) + volume MA(20) + Alligator Jaw(13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_vol_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in high volatility regime
            if high_vol_aligned[i] > 0.5:  # True (1.0) regime
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] > jaw[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] < jaw[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dATR_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0