#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray Bear Power with volume confirmation
# Long when: Alligator bullish alignment (jaw < teeth < lips) AND Elder Ray Bear Power < 0 (bullish) AND volume > 1.3x 20-bar avg
# Short when: Alligator bearish alignment (jaw > teeth > lips) AND Elder Ray Bull Power > 0 (bearish) AND volume > 1.3x 20-bar avg
# Uses 1d ATR for volatility filter to avoid whipsaws in low volatility
# Designed for low trade frequency (15-30/year) with strong trend-following edge in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Williams Alligator (13,8,5) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # === 1d Indicators: Elder Ray (EMA13) and ATR for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Volatility filter: avoid low ATR environments (choppy, low momentum)
        atr_ma_20 = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_1d_aligned[i] > (atr_ma_20[i] * 0.8)  # Only trade when volatility is above 80% of MA
        
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips (mouth opening up)
        # 2. Elder Ray Bear Power < 0 (indicates bullish momentum)
        # 3. Volume confirmation
        # 4. Sufficient volatility
        if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]) and \
           (bear_power_aligned[i] < 0) and \
           vol_confirm and \
           vol_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips (mouth opening down)
        # 2. Elder Ray Bull Power > 0 (indicates bearish momentum)
        # 3. Volume confirmation
        # 4. Sufficient volatility
        elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]) and \
             (bull_power_aligned[i] > 0) and \
             vol_confirm and \
             vol_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Alligator_ElderRay_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0