#!/usr/bin/env python3
"""
6h_PriceAction_SwingRejection_Momentum
Hypothesis: Capture momentum bursts after price rejection at 12h swing points in trending markets.
Works in bull/bear by trading with 12h trend and entering only after rejection candles show
institutional interest. Low frequency via strict swing confirmation and momentum filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for swing points and trend
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 12h swing highs/lows (fractal-like: need confirmation)
    highs_12h = df_12h['high'].values
    lows_12h = df_12h['low'].values
    swing_high = np.zeros_like(highs_12h, dtype=bool)
    swing_low = np.zeros_like(lows_12h, dtype=bool)
    # Swing high: high > left and right, confirmed by next candle close
    for i in range(2, len(highs_12h)-2):
        if (highs_12h[i] > highs_12h[i-1] and highs_12h[i] > highs_12h[i-2] and
            highs_12h[i] > highs_12h[i+1] and highs_12h[i] > highs_12h[i+2]):
            swing_high[i] = True
    # Swing low: low < left and right, confirmed by next candle close
    for i in range(2, len(lows_12h)-2):
        if (lows_12h[i] < lows_12h[i-1] and lows_12h[i] < lows_12h[i-2] and
            lows_12h[i] < lows_12h[i+1] and lows_12h[i] < lows_12h[i+2]):
            swing_low[i] = True
    # Need 2-bar confirmation for swing points (price must close beyond swing level)
    swing_high_confirmed = np.zeros_like(swing_high, dtype=bool)
    swing_low_confirmed = np.zeros_like(swing_low, dtype=bool)
    for i in range(2, len(highs_12h)-2):
        if swing_high[i] and i+2 < len(highs_12h):
            # Confirm if price closes below swing high within 2 bars
            if np.any(highs_12h[i+1:i+3] < highs_12h[i]):
                swing_high_confirmed[i] = True
    for i in range(2, len(lows_12h)-2):
        if swing_low[i] and i+2 < len(lows_12h):
            # Confirm if price closes above swing low within 2 bars
            if np.any(lows_12h[i+1:i+3] > lows_12h[i]):
                swing_low_confirmed[i] = True
    # Align 12h data to 6h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    swing_high_aligned = align_htf_to_ltf(prices, df_12h, swing_high_confirmed.astype(float), additional_delay_bars=2)
    swing_low_aligned = align_htf_to_ltf(prices, df_12h, swing_low_confirmed.astype(float), additional_delay_bars=2)
    
    # 60-period volume average for surge detection
    vol_ma_60 = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    volume_surge = volume > (vol_ma_60 * 2.0)
    
    # Momentum: 5-period ROC > 0
    roc_5 = np.zeros(n)
    for i in range(5, n):
        if close[i-5] != 0:
            roc_5[i] = (close[i] - close[i-5]) / close[i-5]
    momentum = roc_5 > 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or np.isnan(volume_surge[i]) or np.isnan(momentum[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        trend_up = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] if i > 0 else False
        trend_down = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] if i > 0 else False
        
        # Long setup: price rejects swing high in uptrend, then breaks above with momentum
        long_setup = swing_high_aligned[i] > 0 and trend_up
        long_trigger = (close[i] > high[i-1] and  # Break above prior bar high
                       volume_surge[i] and 
                       momentum[i])
        
        # Short setup: price rejects swing low in downtrend, then breaks below with momentum
        short_setup = swing_low_aligned[i] > 0 and trend_down
        short_trigger = (close[i] < low[i-1] and  # Break below prior bar low
                        volume_surge[i] and 
                        momentum[i])
        
        # Exit on opposite swing rejection
        long_exit = swing_low_aligned[i] > 0 and trend_down
        short_exit = swing_high_aligned[i] > 0 and trend_up
        
        if long_setup and long_trigger and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and short_trigger and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_PriceAction_SwingRejection_Momentum"
timeframe = "6h"
leverage = 1.0