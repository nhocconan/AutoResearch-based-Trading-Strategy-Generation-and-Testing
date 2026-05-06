#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily VWAP as dynamic support/resistance with
# volume-weighted trend confirmation and volume spike filter. Long when price
# crosses above VWAP with rising volume and bullish volume trend; short when
# price crosses below VWAP with rising volume and bearish volume trend.
# VWAP acts as institutional reference point, volume confirms institutional
# participation, and volume trend filter ensures momentum alignment.
# Works in bull/bear markets by capturing institutional flow shifts.
# Target: 25-50 trades per year (100-200 over 4 years) with 0.25 position sizing.

name = "4h_dailyVWAP_VolumeTrend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Typical price for VWAP calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # VWAP = cumulative(typical_price * volume) / cumulative(volume)
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align daily VWAP to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Volume trend: 20-period volume moving average slope
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_slope = np.diff(vol_ma, prepend=vol_ma[0])
    vol_trend_bull = vol_ma_slope > 0
    vol_trend_bear = vol_ma_slope < 0
    
    # Volume spike: current volume > 1.8x 20-period average
    volume_spike = volume > (1.8 * vol_ma)
    
    # Price relative to VWAP for crossover detection
    price_above_vwap = close > vwap_aligned
    price_below_vwap = close < vwap_aligned
    
    # Detect VWAP crossovers
    vwap_cross_up = np.zeros(n, dtype=bool)
    vwap_cross_down = np.zeros(n, dtype=bool)
    vwap_cross_up[1:] = price_above_vwap[1:] & ~price_above_vwap[:-1]
    vwap_cross_down[1:] = price_below_vwap[1:] & ~price_below_vwap[:-1]
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VWAP cross up with volume spike and bullish volume trend
            if vwap_cross_up[i] and volume_spike[i] and vol_trend_bull[i]:
                signals[i] = 0.25
                position = 1
            # Short: VWAP cross down with volume spike and bearish volume trend
            elif vwap_cross_down[i] and volume_spike[i] and vol_trend_bear[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below VWAP (trend reversal)
            if vwap_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above VWAP (trend reversal)
            if vwap_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals