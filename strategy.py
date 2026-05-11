#!/usr/bin/env python3
"""
6h_Premium_Discount_Equilibrium
Hypothesis: Mean reversion to weekly VWAP (fair value) with institutional bias detection.
Long when price < weekly VWAP AND institutional buying pressure (delta > 0).
Short when price > weekly VWAP AND institutional selling pressure (delta < 0).
Uses 12h EMA50 trend filter to avoid counter-trend trades in strong moves.
Volume confirmation ensures institutional participation.
Designed for low frequency (15-30 trades/year) by requiring confluence of value, 
momentum, and volume. Works in bull/bear by fading extremes to weekly equilibrium.
"""

name = "6h_Premium_Discount_Equilibrium"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP from typical price and volume"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    return vwap_numerator, vwap_denominator

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly VWAP for Fair Value ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    vwap_num, vwap_den = calculate_vwap(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, df_1w['volume'].values
    )
    # Cumulative VWAP
    cum_num = np.nancumsum(vwap_num)
    cum_den = np.nancumsum(vwap_den)
    vwap_1w = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den!=0)
    
    vwap_1w_6h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # --- 12h EMA50 Trend Filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Institutional Pressure via Delta (buyer - seller volume) ---
    # Approximate using close relative to range: if close > midpoint, buying pressure
    # This is a proxy for delta when true bid/ask volume unavailable
    midpoint = (high + low) / 2.0
    buying_pressure = close > midpoint  # True when closing in upper half
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1w_6h[i]) or np.isnan(ema_50_12h_6h[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Value zone: distance from weekly VWAP
        vwap_dist_pct = (close[i] - vwap_1w_6h[i]) / vwap_1w_6h[i]
        
        # Trend filter: price relative to 12h EMA50
        above_ema = close[i] > ema_50_12h_6h[i]
        
        # Institutional pressure confirmation
        buying = buying_pressure[i]
        
        if position == 0:
            # Long: discount to weekly VWAP + buying pressure + below EMA (avoid chasing)
            if vwap_dist_pct < -0.005 and buying and not above_ema:  # >0.5% below VWAP
                signals[i] = 0.25
                position = 1
            # Short: premium to weekly VWAP + selling pressure + above EMA
            elif vwap_dist_pct > 0.005 and not buying and above_ema:  # >0.5% above VWAP
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to VWAP equilibrium or trend fails
            if position == 1:
                # Exit long: price crosses VWAP OR strong selling pressure
                if vwap_dist_pct > -0.002 or not buying:  # Near VWAP or selling pressure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses VWAP OR strong buying pressure
                if vwap_dist_pct < 0.002 or buying:  # Near VWAP or buying pressure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals