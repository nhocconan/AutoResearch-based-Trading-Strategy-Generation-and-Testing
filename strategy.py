#!/usr/bin/env python3
"""
12h_rsi_pullback_1d_trend_v2
Hypothesis: On 12h timeframe, buy pullbacks in uptrend (identified by 1d EMA200) and sell rallies in downtrend, using RSI(14) for entry timing. Volume confirmation filters low-participation moves. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while capturing trend continuation moves in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_pullback_1d_trend_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 12h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(200, 20), n):
        # Skip if data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns bearish
            if rsi_values[i] > 70 or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns bullish
            if rsi_values[i] < 30 or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # In uptrend: buy RSI pullback from overbought
                if close[i] > ema200_1d_aligned[i] and rsi_values[i] < 40 and rsi_values[i-1] >= 40:
                    position = 1
                    signals[i] = 0.25
                # In downtrend: sell RSI bounce from oversold
                elif close[i] < ema200_1d_aligned[i] and rsi_values[i] > 60 and rsi_values[i-1] <= 60:
                    position = -1
                    signals[i] = -0.25
    
    return signals