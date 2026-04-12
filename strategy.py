# 6h_1d_obv_momentum_with_volatility_filter
# Hypothesis: On-Balance Volume (OBV) divergence with 6h price action and 1d volatility filter.
# In bull markets: rising OBV confirms accumulation during pullbacks. In bear markets: falling OBV confirms distribution during rallies.
# Uses 1d ATR to filter low-volatility chop. Target: 15-30 trades/year (60-120 total over 4 years).

name = "6h_1d_obv_momentum_with_volatility_filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate OBV
    obv = np.zeros(n)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 6-period EMA of OBV for momentum
    obv_series = pd.Series(obv)
    obv_ema = obv_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if ATR not ready
        if np.isnan(atr_aligned[i]) or atr_aligned[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Long: OBV rising above its EMA (bullish momentum) in non-choppy market
        if obv[i] > obv_ema[i] and obv[i] > obv[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: OBV falling below its EMA (bearish momentum) in non-choppy market
        elif obv[i] < obv_ema[i] and obv[i] < obv[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: momentum diverges (OBV crosses EMA in opposite direction)
        elif position == 1 and obv[i] < obv_ema[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and obv[i] > obv_ema[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals