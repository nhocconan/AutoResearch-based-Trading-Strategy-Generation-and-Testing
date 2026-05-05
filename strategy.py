#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d ATR regime filter + volume confirmation
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5x 20-period MA
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND ATR(14) > ATR(50) AND volume > 1.5x 20-period MA
# Exit when: Alligator alignment breaks OR ATR(14) < ATR(50) (low volatility regime)
# Uses Alligator for trend direction, ATR regime for volatility filter, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsAlligator_1dATRRegime_VolumeConfirm"
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
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Williams Alligator on 12h
    # Jaw (blue line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (red line): 8-period SMMA smoothed 5 periods ahead  
    # Lips (green line): 5-period SMMA smoothed 3 periods ahead
    if len(close) >= 13:
        # SMMA calculation (smoothed moving average)
        def smma(data, period):
            if len(data) < period:
                return np.full(len(data), np.nan)
            result = np.full(len(data), np.nan)
            # First value is SMA
            result[period-1] = np.mean(data[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        jaw = smma(close, 13)
        teeth = smma(close, 8)
        lips = smma(close, 5)
        
        # Alligator alignment: Jaw < Teeth < Lips = bullish, Jaw > Teeth > Lips = bearish
        bullish_alignment = (jaw < teeth) & (teeth < lips)
        bearish_alignment = (jaw > teeth) & (teeth > lips)
    else:
        jaw = np.full(n, np.nan)
        teeth = np.full(n, np.nan)
        lips = np.full(n, np.nan)
        bullish_alignment = np.zeros(n, dtype=bool)
        bearish_alignment = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for ATR(50)
        return np.zeros(n)
    
    # Calculate ATR on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr1[0] = 0  # first period has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR calculation
        def atr(data, period):
            if len(data) < period:
                return np.full(len(data), np.nan)
            result = np.full(len(data), np.nan)
            # First value is ATR = average of first 'period' TR values
            result[period-1] = np.mean(data[:period])
            # Subsequent values: ATR = (PREV_ATR * (period-1) + CURRENT_TR) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr_14 = atr(tr, 14)
        atr_50 = atr(tr, 50)
        
        # ATR regime: ATR(14) > ATR(50) = high volatility regime (good for trend following)
        atr_regime = atr_14 > atr_50
    else:
        atr_14 = np.full(len(df_1d), np.nan)
        atr_50 = np.full(len(df_1d), np.nan)
        atr_regime = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d ATR regime to 12h timeframe
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment + ATR regime + volume filter
            if (bullish_alignment[i] and 
                atr_regime_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment + ATR regime + volume filter
            elif (bearish_alignment[i] and 
                  atr_regime_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR ATR regime changes to low volatility
            if not (bullish_alignment[i] and atr_regime_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR ATR regime changes to low volatility
            if not (bearish_alignment[i] and atr_regime_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals