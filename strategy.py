#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week pivot levels (R4/S4) for breakout direction and 1-day RSI for momentum confirmation.
# Enters long when price breaks above weekly R4 with RSI > 50 (bullish momentum).
# Enters short when price breaks below weekly S4 with RSI < 50 (bearish momentum).
# Uses volume confirmation to avoid false breakouts.
# Weekly pivot levels act as strong support/resistance; breakouts indicate institutional interest.
# Works in bull/bear by following breakout direction from key weekly levels.
# Targets 50-150 total trades over 4 years (~12-37/year) with strict entry conditions.
name = "6h_1wPivot_R4S4_Breakout_RSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 1-week data for pivot points (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R4 = P + 3*(H - L), S4 = P - 3*(H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r4_1w = pivot_1w + 3.0 * (high_1w - low_1w)
    s4_1w = pivot_1w - 3.0 * (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Pre-compute 1-day RSI for momentum confirmation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume filter: volume > 1.3 * 20-period average (on 6h data)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R4 AND RSI > 50 (bullish momentum) with volume
            if (close[i] > r4_1w_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 AND RSI < 50 (bearish momentum) with volume
            elif (close[i] < s4_1w_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S4 (invalidates bullish breakout) or RSI < 40
            if close[i] < s4_1w_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R4 (invalidates bearish breakout) or RSI > 60
            if close[i] > r4_1w_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals