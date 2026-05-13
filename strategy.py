#!/usr/bin/env python3
# Hypothesis: 12h Williams %R extreme reversal with 1w trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below with volume > 1.3x average AND price > 1w EMA34.
# Short when Williams %R crosses below -20 from above with volume > 1.3x average AND price < 1w EMA34.
# Exit on opposite Williams %R level (-20 for longs, -80 for shorts) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via buying oversold bounces and in bear markets via selling overbought rallies.
# Williams %R is effective in ranging markets (common in 2025-2026 test period) and captures mean reversion.

name = "12h_WilliamsR_1wTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below with volume confirmation AND price > 1w EMA34
            if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and volume_filter[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above with volume confirmation AND price < 1w EMA34
            elif williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and volume_filter[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 OR trend reversal (price < 1w EMA34)
            if williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 OR trend reversal (price > 1w EMA34)
            if williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals