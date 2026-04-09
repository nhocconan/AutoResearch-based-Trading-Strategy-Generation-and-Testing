#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Filters by 1d ADX(14) > 25 to ensure we trade only in trending markets
# - In strong trends, extreme Williams %R readings often precede continuation rather than reversal
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)

name = "6h_1d_williamsr_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    plus_di_1d = 100 * WilderSmoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * WilderSmoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = WilderSmoothing(dx_1d, 14)
    
    # 1d ADX > 25 indicates strong trend
    strong_trend = adx_1d > 25
    
    # Align 1d ADX trend filter to 6h
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(strong_trend_aligned[i]) or
            strong_trend_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading) or reverse signal
            if williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought - reverse to short
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading) or reverse signal
            if williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold - reverse to long
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when oversold in strong trend
            if williams_r[i] < -80:
                position = 1
                signals[i] = 0.25
            # Enter short when overbought in strong trend
            elif williams_r[i] > -20:
                position = -1
                signals[i] = -0.25
    
    return signals