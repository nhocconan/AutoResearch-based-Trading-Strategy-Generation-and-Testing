#!/usr/bin/env python3
# 6h_Liquidity_Imbalance_Reversal_12hTrend
# Hypothesis: Trade reversals at liquidity imbalances (volume-weighted price gaps) aligned with 12h trend.
# In bull markets: buy dips to unfilled buy-side liquidity (below VWAP, high volume imbalance).
# In bear markets: sell rallies to unfilled sell-side liquidity (above VWAP, high volume imbalance).
# Uses 12h EMA50 for trend filter and 6h volume-weighted imbalance for entry timing.
# Designed for low trade frequency (15-25/year) to minimize fee drag in ranging/bear markets.

name = "6h_Liquidity_Imbalance_Reversal_12hTrend"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 6h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Calculate VWAP and volume imbalance for 6h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Volume delta (buying vs selling pressure approximation)
    # Using close proximity to high/low as proxy for aggression
    close_position = (close - low) / (high - low + 1e-10)  # 0 at low, 1 at high
    buying_pressure = close_position * volume
    selling_pressure = (1 - close_position) * volume
    volume_imbalance = buying_pressure - selling_pressure  # +ve = buying pressure
    
    # Identify significant liquidity imbalances (unfilled orders)
    # Look for divergence: price moving against volume pressure with high volume
    price_change = np.diff(close, prepend=close[0])
    volume_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ratio = volume / (volume_ma + 1e-10)
    
    # Liquidity imbalance signals: price drops on high buying volume (bullish imbalance)
    # or price rises on high selling volume (bearish imbalance)
    bullish_imbalance = (price_change < 0) & (volume_imbalance > 0) & (volume_ratio > 1.5)
    bearish_imbalance = (price_change > 0) & (volume_imbalance < 0) & (volume_ratio > 1.5)
    
    # Distance from VWAP for entry quality
    vwap_distance_bull = (vwap - low) / vwap  # How far below VWAP the low went
    vwap_distance_bear = (high - vwap) / vwap  # How far above VWAP the high went
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(vwap[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish liquidity imbalance (price dipped on buying volume) in uptrend
            # Price came significantly below VWAP on buying pressure
            if (bullish_imbalance[i] and
                vwap_distance_bull[i] > 0.003 and  # At least 0.3% below VWAP
                trend_12h_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: bearish liquidity imbalance (price rallied on selling volume) in downtrend
            elif (bearish_imbalance[i] and
                  vwap_distance_bear[i] > 0.003 and  # At least 0.3% above VWAP
                  trend_12h_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to VWAP (liquidity filled) or trend breaks
            if (close[i] >= vwap[i] or  # Price reached VWAP
                trend_12h_up_aligned[i] < 0.5):  # Trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to VWAP (liquidity filled) or trend breaks
            if (close[i] <= vwap[i] or  # Price reached VWAP
                trend_12h_down_aligned[i] < 0.5):  # Trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals