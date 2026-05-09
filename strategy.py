# #!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation
# Long when: Bollinger Band width at 20-period low + price breaks above upper band + 12h EMA50 up + volume spike
# Short when: Bollinger Band width at 20-period low + price breaks below lower band + 12h EMA50 down + volume spike
# Exit when price returns to middle band or 12h EMA direction reverses
# Uses Bollinger Band width to identify low volatility periods before breakouts
# Works in both bull and bear markets by following 12h trend direction
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_Bollinger_Squeeze_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    bband_width = upper - lower
    
    # Bollinger Band width percentile (20-period lookback)
    bband_width_series = pd.Series(bband_width.values)
    width_percentile = bband_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    ema12_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema12_50_aligned = align_htf_to_ltf(prices, df_12h, ema12_50)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(basis[i]) or
            np.isnan(width_percentile[i]) or np.isnan(ema12_50_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze: width at 20-period low (< 20th percentile)
            is_squeeze = width_percentile[i] <= 0.20
            
            # Enter long: squeeze + break above upper band + 12h EMA up + volume spike
            if (is_squeeze and 
                close[i] > upper[i] and 
                ema12_50_aligned[i] > ema12_50_aligned[i-1] and  # 12h EMA rising
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + break below lower band + 12h EMA down + volume spike
            elif (is_squeeze and 
                  close[i] < lower[i] and 
                  ema12_50_aligned[i] < ema12_50_aligned[i-1] and  # 12h EMA falling
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band OR 12h EMA turns down
            if (close[i] < basis[i]) or (ema12_50_aligned[i] < ema12_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle band OR 12h EMA turns up
            if (close[i] > basis[i]) or (ema12_50_aligned[i] > ema12_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals