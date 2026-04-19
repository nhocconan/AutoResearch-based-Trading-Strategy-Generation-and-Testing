# 1h_Pivot_R1_S1_Breakout_Volume_Session
# Hypothesis: Use daily Pivot (R1/S1) for directional bias, 1h for entry timing with volume confirmation.
# Trade only during active hours (08-20 UTC) to reduce noise. Target 15-37 trades/year per symbol.
# Works in bull/bear: Pivot levels act as support/resistance; breakouts with volume indicate institutional interest.
# Uses daily P/R1/S1 as structural levels, avoiding overtrading via session filter and volume confirmation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Pivot_R1_S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Pivot points for structural bias
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Pivot, R1, S1
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align daily levels to 1h timeframe (uses previous day's close)
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current > 1.5x 24-period average (1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(vol_ma_24[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume confirmation
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_1h[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume
            elif price < s1_1h[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: return to S1 (mean reversion to opposite level)
            if price < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: return to R1 (mean reversion to opposite level)
            if price > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals