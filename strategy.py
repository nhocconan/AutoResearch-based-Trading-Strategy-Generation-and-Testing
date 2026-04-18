# 12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# Hypothesis: Camarilla pivot breakout with volume confirmation and ATR volatility filter on 12h timeframe.
# Uses daily pivots as structural support/resistance. Breakouts above R1 or below S1 with volume > 1.5x average
# indicate institutional interest. ATR filter avoids whipsaw in low volatility. Designed for 12h to capture
# multi-day trends while minimizing trades (target: 15-30/year). Works in bull/bear by following breakout direction.
# Long when close > R1 + volume confirmation + ATR > 20-period average.
# Short when close < S1 + volume confirmation + ATR > 20-period average.
# Exit when price reverts to pivot point (PP) or ATR drops below average (low volatility environment).

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + ((high_1d - low_1d) * 1.0833)
    s1 = close_1d - ((high_1d - low_1d) * 1.0833)
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = np.full_like(close, np.nan)
    atr_period = 14
    
    if len(tr) >= atr_period:
        # Initial ATR
        atr[atr_period-1] = np.mean(tr[:atr_period])
        # Wilder smoothing
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Align ATR to 12h (already aligned as calculated on 12h data)
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ATR filter: only trade when volatility is above average
        atr_ma = np.full_like(atr, np.nan)
        if len(atr) >= 20:
            if i >= 20:
                atr_ma[i] = np.mean(atr[i-20:i])
        atr_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 0:
            # Long: close > R1 + volume confirmation + ATR filter
            if close[i] > r1_aligned[i] and vol_confirm and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 + volume confirmation + ATR filter
            elif close[i] < s1_aligned[i] and vol_confirm and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close < PP (reversion to mean) or ATR drops below average (low volatility)
            if close[i] < pp_aligned[i] or (not np.isnan(atr_ma[i]) and atr[i] < atr_ma[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close > PP (reversion to mean) or ATR drops below average (low volatility)
            if close[i] > pp_aligned[i] or (not np.isnan(atr_ma[i]) and atr[i] < atr_ma[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0