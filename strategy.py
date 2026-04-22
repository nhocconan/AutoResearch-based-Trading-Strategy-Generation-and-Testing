#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1-week EMA50 trend filter
# Elder Ray measures bull/bear power relative to EMA13, filtering by weekly trend.
# Works in bull markets (buy when bull power > 0 and weekly uptrend) and bear markets (sell when bear power < 0 and weekly downtrend).
# Uses volume confirmation to avoid false signals. Target: 15-25 trades/year per symbol (60-100 total).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align weekly EMA50 to 6-hour timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive + weekly uptrend + volume confirmation
            if (bull_power[i] > 0 and close[i] > ema50_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative + weekly downtrend + volume confirmation
            elif (bear_power[i] < 0 and close[i] < ema50_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder power contradicts position or volume fails
            if position == 1:
                if (bull_power[i] <= 0 or not vol_confirm[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (bear_power[i] >= 0 or not vol_confirm[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wEMA50_Volume_Session"
timeframe = "6h"
leverage = 1.0