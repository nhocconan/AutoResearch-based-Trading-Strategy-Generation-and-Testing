# 4h Camarilla R1/S1 Breakout with Volume and ATR Filter
# Hypothesis: Combines price channel breakout (Camarilla levels) with volume confirmation
# and ATR-based volatility filter to avoid false breakouts in low volatility periods.
# Works in bull markets by catching breakouts and in bear markets by filtering out
# whipsaws during low volatility. Uses tight entry conditions to limit trades.
# Timeframe: 4h, HTF: 1d for pivots, 12h for trend filter.

#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_Volume_ATR_Filter"
timeframe = "4h"
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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # ATR filter: avoid low volatility periods
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max(tr1[0], tr2[0]) if len(tr1) > 0 else 0], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr > (0.5 * atr_ma)  # Only trade when volatility is above half its 50-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_filter[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 + above 12h EMA50 + volume filter + volatility filter
            if high[i] > r1_1d_aligned[i] and close[i] > ema_50_12h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 + below 12h EMA50 + volume filter + volatility filter
            elif low[i] < s1_1d_aligned[i] and close[i] < ema_50_12h_aligned[i] and vol_filter[i] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below S1 or below 12h EMA50
            if low[i] < s1_1d_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above R1 or above 12h EMA50
            if high[i] > r1_1d_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals