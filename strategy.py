#!/usr/bin/env python3
# 1h_DailyTrend_With_1dVWAP_Support_Resistance
# Hypothesis: Use daily trend direction (price above/below daily VWAP) as primary filter, 
# then enter on 1h pullbacks to 20 EMA during London/NY session (08-20 UTC).
# Daily VWAP acts as dynamic support/resistance - price tends to respect it.
# In bull markets: buy dips to EMA when above daily VWAP.
# In bear markets: sell rallies to EMA when below daily VWAP.
# Session filter reduces noise and focuses on liquid hours.
# Target: 15-30 trades/year by requiring both trend alignment and pullback.

name = "1h_DailyTrend_With_1dVWAP_Support_Resistance"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Daily VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    daily_vwap = (vwap_numerator / vwap_denominator).values
    
    # Align daily VWAP to 1h timeframe (with 1-day delay for completed daily VWAP)
    daily_vwap_aligned = align_htf_to_ltf(prices, df_1d, daily_vwap, additional_delay_bars=1)
    
    # 20 EMA on 1h for pullback entries
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if required data is NaN
        if np.isnan(daily_vwap_aligned[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Determine daily trend: price above/below daily VWAP
        price_above_vwap = close[i] > daily_vwap_aligned[i]
        
        if position == 0:
            # Long: price above daily VWAP (bullish trend) + pullback to EMA20
            if price_above_vwap and close[i] <= ema_20[i] * 1.002:  # Allow small buffer
                signals[i] = 0.20
                position = 1
            # Short: price below daily VWAP (bearish trend) + rally to EMA20
            elif not price_above_vwap and close[i] >= ema_20[i] * 0.998:  # Allow small buffer
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below daily VWAP (trend change) or significant rally
            if not price_above_vwap or close[i] >= ema_20[i] * 1.03:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above daily VWAP (trend change) or significant pullback
            if price_above_vwap or close[i] <= ema_20[i] * 0.97:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals