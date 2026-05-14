#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator trend with volume confirmation and ATR filter
# - Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND volume > 1.5x 20-period average AND 1d ATR(14) < 20-period median ATR
# - Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND volume > 1.5x 20-period average AND 1d ATR(14) < 20-period median ATR
# - Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order) OR price crosses back below/above teeth
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Williams Alligator identifies trending markets with smoothed moving averages
# - Volume confirmation reduces false breakouts
# - ATR filter ensures we trade during low volatility periods when trends are more reliable

name = "4h_1d_alligator_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
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
    
    # Pre-compute 4h Williams Alligator (Smoothed Moving Averages)
    # Alligator: Jaw (13-period SMMA, shifted 8 bars), Teeth (8-period SMMA, shifted 5 bars), Lips (5-period SMMA, shifted 3 bars)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
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
            # Bullish Alligator alignment: Jaw < Teeth < Lips
            bullish_alignment = jaw[i] < teeth[i] < lips[i]
            # Bearish Alligator alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw[i] > teeth[i] > lips[i]
            
            # Long conditions: bullish alignment AND price > lips AND low volatility regime AND volume spike
            if (bullish_alignment and 
                close[i] > lips[i] and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND price < lips AND low volatility regime AND volume spike
            elif (bearish_alignment and 
                  close[i] < lips[i] and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator alignment breaks OR price crosses back below/above teeth
            bullish_alignment = jaw[i] < teeth[i] < lips[i]
            bearish_alignment = jaw[i] > teeth[i] > lips[i]
            
            exit_long = (position == 1 and (not bullish_alignment or close[i] < teeth[i]))
            exit_short = (position == -1 and (not bearish_alignment or close[i] > teeth[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals