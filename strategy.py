#/usr/bin/env python3
"""
1d_Weekly_Trix_Trend_With_Volume_Spike
Hypothesis: TRIX on weekly timeframe captures long-term momentum. 
When weekly TRIX crosses above zero with daily volume spike and price above daily EMA(50), 
go long. When weekly TRIX crosses below zero with volume spike and price below daily EMA(50), 
go short. Uses 0.25 position size to limit trade frequency (~10-25/year) and reduce fee drag.
Works in bull markets via momentum continuation and in bear markets via mean reversion 
when momentum extremes coincide with volume exhaustion.
"""

name = "1d_Weekly_Trix_Trend_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for TRIX (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate TRIX: triple EMA of percent change
    # ROC = (close / close.shift(1) - 1) * 100
    roc = (df_weekly['close'] / df_weekly['close'].shift(1) - 1) * 100
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 of EMA2 = TRIX
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3
    
    # Align TRIX to daily timeframe
    trix_aligned = align_htf_to_ltf(prices, df_weekly, trix)
    
    # Daily EMA(50) for trend filter
    ema50_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Weekly TRIX crosses above zero with volume spike and uptrend
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                volume_spike[i] and 
                close[i] > ema50_daily[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly TRIX crosses below zero with volume spike and downtrend
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  volume_spike[i] and 
                  close[i] < ema50_daily[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly TRIX crosses below zero or price breaks EMA50
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               (close[i] < ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly TRIX crosses above zero or price breaks EMA50
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               (close[i] > ema50_daily[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals