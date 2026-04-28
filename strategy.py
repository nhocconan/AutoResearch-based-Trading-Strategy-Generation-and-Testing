#!/usr/bin/env python3
"""
6h_HeikinAshi_Trend_12hVWAP_Reversion
Hypothesis: Heikin Ashi candles filter noise to capture true 6h trends, while 12h VWAP acts as dynamic mean reversion level. 
In bull markets: buy HA green candles above 12h VWAP. In bear markets: sell HA red candles below 12h VWAP.
Uses volume-weighted average price as institutional reference point. Targets 15-30 trades/year by requiring HA trend confirmation + VWAP deviation.
Works in both bull (trend following) and bear (mean reversion from VWAP) regimes.
"""

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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate Heikin Ashi close: (O+H+L+C)/4
    ha_close = (open_price + high + low + close) / 4
    # Calculate Heikin Ashi open: (prev HA open + prev HA close)/2
    ha_open = np.zeros(n)
    ha_open[0] = open_price[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    # HA high/low for completeness
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # HA trend: green candle (close > open) = bullish, red candle = bearish
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for 12h: cumulative (price * volume) / cumulative volume
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vp_12h = typical_price_12h * df_12h['volume']
    cum_vp_12h = vp_12h.cumsum()
    cum_vol_12h = df_12h['volume'].cumsum()
    vwap_12h = cum_vp_12h / cum_vol_12h
    
    # Align 12h VWAP to 6h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h.values)
    
    # Deviation from VWAP: percent distance
    vwap_deviation = (close - vwap_12h_aligned) / vwap_12h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(ha_bullish[i]) or np.isnan(ha_bearish[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: HA bullish + price above VWAP + volume surge
        long_condition = ha_bullish[i] and (vwap_deviation[i] > 0) and volume_surge[i]
        # Short conditions: HA bearish + price below VWAP + volume surge
        short_condition = ha_bearish[i] and (vwap_deviation[i] < 0) and volume_surge[i]
        
        # Exit conditions: HA color change or VWAP cross
        exit_long = ha_bearish[i] or (vwap_deviation[i] < -0.005)  # 0.5% below VWAP
        exit_short = ha_bullish[i] or (vwap_deviation[i] > 0.005)   # 0.5% above VWAP
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_long and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif exit_short and position == -1:
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

name = "6h_HeikinAshi_Trend_12hVWAP_Reversion"
timeframe = "6h"
leverage = 1.0