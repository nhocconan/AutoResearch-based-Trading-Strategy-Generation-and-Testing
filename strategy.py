#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ATR regime filter and volume confirmation
# - Uses Williams Alligator (jaw=13, teeth=8, lips=5) to identify trending vs ranging markets
# - Long when: price > Alligator teeth AND Alligator is aligned (jaw > teeth > lips) AND 1d ATR < median ATR AND volume > 1.5x avg volume
# - Short when: price < Alligator teeth AND Alligator is aligned (jaw < teeth < lips) AND 1d ATR < median ATR AND volume > 1.5x avg volume
# - Exit when price crosses Alligator teeth or Alligator loses alignment
# - Discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Williams Alligator identifies trend strength and direction with built-in smoothing
# - ATR filter ensures we trade during low volatility periods when trends are more reliable
# - Volume confirmation reduces false breakouts

name = "4h_1d_alligator_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator (jaw=13, teeth=8, lips=5) - using SMMA (smoothed moving average)
    def smma(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < window:
            return result
        # First value is simple SMA
        result[window-1] = np.mean(arr[:window])
        # Subsequent values are smoothed
        for i in range(window, len(arr)):
            result[i] = (result[i-1] * (window-1) + arr[i]) / window
        return result
    
    # Alligator lines
    jaw = smma(high, 13)  # Jaw (blue) - 13-period SMMA of median price
    teeth = smma(high, 8)  # Teeth (red) - 8-period SMMA of median price
    lips = smma(high, 5)   # Lips (green) - 5-period SMMA of median price
    
    # Use median price for Alligator calculation (more representative)
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Alligator alignment: jaw > teeth > lips for uptrend, jaw < teeth < lips for downtrend
    jaw_gt_teeth = jaw > teeth
    teeth_gt_lips = teeth > lips
    jaw_lt_teeth = jaw < teeth
    teeth_lt_lips = teeth < lips
    alligator_align_up = jaw_gt_teeth & teeth_gt_lips
    alligator_align_down = jaw_lt_teeth & teeth_lt_lips
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_median_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_1d < atr_median_20
    
    # Align HTF indicators to 4h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price > teeth AND alligator aligned up AND low volatility regime AND volume spike
            if (close[i] > teeth[i] and 
                alligator_align_up[i] and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < teeth AND alligator aligned down AND low volatility regime AND volume spike
            elif (close[i] < teeth[i] and 
                  alligator_align_down[i] and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses teeth or alligator loses alignment
            exit_long = (position == 1 and (close[i] < teeth[i] or not alligator_align_up[i]))
            exit_short = (position == -1 and (close[i] > teeth[i] or not alligator_align_down[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals