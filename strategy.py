#!/usr/bin/env python3
"""
1d Daily ATR Breakout with 1W Trend and Volume Confirmation
Hypothesis: Daily ATR breakouts capture momentum moves, filtered by weekly trend (EMA20) to avoid counter-trend trades and volume confirmation to avoid false signals. Works in bull/bear by aligning with higher timeframe trend. Targets 15-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_atr_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = df_1w['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily ATR(14) for volatility and breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (>1.5x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after ATR warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below open - ATR*1.5 (trailing stop)
            if close[i] <= (prices['open'].iloc[i] - 1.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above open + ATR*1.5 (trailing stop)
            if close[i] >= (prices['open'].iloc[i] + 1.5 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above close + ATR*2, uptrend, volume
            if (close[i] > (prices['close'].iloc[i-1] + 2 * atr[i]) and 
                close[i] > ema_20_1w_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below close - ATR*2, downtrend, volume
            elif (close[i] < (prices['close'].iloc[i-1] - 2 * atr[i]) and 
                  close[i] < ema_20_1w_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals