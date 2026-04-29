#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ATR Regime Filter + Volume Spike
# Long when: Jaw > Teeth > Lips (bullish alignment) AND ATR(14) > ATR(50) (high volatility regime) AND Volume > 1.5 * Volume MA(20)
# Short when: Jaw < Teeth < Lips (bearish alignment) AND ATR(14) > ATR(50) AND Volume > 1.5 * Volume MA(20)
# Uses Williams Alligator for trend identification, ATR regime to avoid low-volatility whipsaws, volume spike for confirmation.
# Timeframe: 12h (primary), HTF: 1d for ATR regime filter.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.

name = "12h_WilliamsAlligator_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for ATR regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_close[0] = np.nan
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - prev_close)
    tr3 = np.abs(df_1d['low'].values - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ATR(14) > ATR(50) indicates high volatility regime
    atr_regime = atr_14 > atr_50
    
    # Align ATR regime to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Calculate Williams Alligator on 12h data
    # Jaw (Blue Line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red Line): 8-period SMMA smoothed 5 periods ahead
    # Lips (Green Line): 5-period SMMA smoothed 3 periods ahead
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(arr, period):
        """Calculate Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift Jaw, Teeth, Lips forward by their respective offsets
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Invalidate the shifted values that don't have enough history
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume spike filter on 12h: Volume > 1.5 * Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 8, 5, 3, 20)  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if ATR regime data not available
        if np.isnan(atr_regime_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_atr_regime = atr_regime_aligned[i] > 0.5  # Convert back to boolean
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (Jaw <= Teeth or Teeth <= Lips)
            # 2. Low volatility regime (ATR(14) <= ATR(50))
            if (curr_jaw <= curr_teeth or curr_teeth <= curr_lips or not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (Jaw >= Teeth or Teeth >= Lips)
            # 2. Low volatility regime (ATR(14) <= ATR(50))
            if (curr_jaw >= curr_teeth or curr_teeth >= curr_lips or not curr_atr_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish Alligator alignment: Jaw > Teeth > Lips
            bullish_alignment = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
            # Bearish Alligator alignment: Jaw < Teeth < Lips
            bearish_alignment = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
            
            # Long entry: bullish alignment AND high volatility regime AND volume spike
            if bullish_alignment and curr_atr_regime and curr_volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment AND high volatility regime AND volume spike
            elif bearish_alignment and curr_atr_regime and curr_volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals