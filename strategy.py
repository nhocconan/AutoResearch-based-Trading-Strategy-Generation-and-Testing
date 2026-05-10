# 6h_Angel_Rebound_Strategy
# Hypothesis: Buy on 6h pullbacks to 1d EMA200 during strong uptrends, sell rallies to 1d EMA200 during strong downtrends.
# Uses 1d EMA200 as dynamic support/resistance with 60-period 6h EMA for trend confirmation.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 20-40 trades/year (80-160 total over 4 years) to stay within optimal trade frequency for 6h.

name = "6h_Angel_Rebound_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA200 for dynamic support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 60-period 6h EMA for trend confirmation (bullish above, bearish below)
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average (reduces false signals)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_60[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near 1d EMA200 support, 6h EMA bullish, volume confirmation
            if (low[i] <= ema_200_1d_aligned[i] * 1.005 and  # within 0.5% above EMA200
                close[i] > ema_60[i] and                     # 6h EMA bullish
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near 1d EMA200 resistance, 6h EMA bearish, volume confirmation
            elif (high[i] >= ema_200_1d_aligned[i] * 0.995 and  # within 0.5% below EMA200
                  close[i] < ema_60[i] and                     # 6h EMA bearish
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price moves significantly above 1d EMA200 or trend turns bearish
            if (close[i] > ema_200_1d_aligned[i] * 1.02 or  # 2% above EMA200
                close[i] < ema_60[i]):                       # trend turned bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price moves significantly below 1d EMA200 or trend turns bullish
            if (close[i] < ema_200_1d_aligned[i] * 0.98 or  # 2% below EMA200
                close[i] > ema_60[i]):                       # trend turned bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals