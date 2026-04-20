# 6h_VWAP_Cross_1dTrend_Retest
# Hypothesis: VWAP cross on 6h combined with 1d trend filter and pullback retest.
# Uses VWAP as mean reversion anchor; trades only in direction of daily trend.
# Retest of VWAP after breakout reduces false signals. Works in both bull/bear via trend filter.
# Target: 20-50 trades/year, conservative size 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA(50) for trend regime
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h VWAP calculation (typical price * volume / cumulative volume)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    tpv = typical_price * prices['volume']
    cum_tpv = tpv.cumsum()
    cum_vol = prices['volume'].cumsum()
    vwap = cum_tpv / cum_vol
    vwap = vwap.values  # numpy array
    
    # Price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above VWAP, daily uptrend, and retest condition (pullback to VWAP)
            crossed_above = price > vwap_val and close[i-1] <= vwap[i-1]
            retest = abs(price - vwap_val) < (vwap_val * 0.001)  # within 0.1% of VWAP
            if ema_50_val < price and crossed_above and retest:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP, daily downtrend, and retest condition
            crossed_below = price < vwap_val and close[i-1] >= vwap[i-1]
            if ema_50_val > price and crossed_below and retest:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below VWAP or daily trend turns down
            if price < vwap_val or ema_50_val > price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above VWAP or daily trend turns up
            if price > vwap_val or ema_50_val < price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_Cross_1dTrend_Retest"
timeframe = "6h"
leverage = 1.0