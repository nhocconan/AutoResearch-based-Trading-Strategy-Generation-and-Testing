#!/usr/bin/env python3
"""
6h_OrderBlock_Equilibrium_Rebalance
Hypothesis: Price revisits institutional order blocks (equilibrium zones) after strong moves.
Institutions leave unfilled orders at swing points; price returns to rebalance before continuing.
Long when price retraces to bullish OB in uptrend; short when price retraces to bearish OB in downtrend.
Uses 12h trend filter and volume confirmation to avoid false signals in ranging markets.
Works in bull (buy dips) and bear (sell rallies) by trading mean reversion within trend.
Target: 15-30 trades/year per symbol.
"""

name = "6h_OrderBlock_Equilibrium_Rebalance"
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
    
    # Identify swing points: pivot highs/lows (3-bar lookback)
    # Bullish OB: base before strong up move (lowest low of 3 bars before bullish break)
    # Bearish OB: base before strong down move (highest high of 3 bars before bearish break)
    bullish_ob = np.full(n, np.nan)
    bearish_ob = np.full(n, np.nan)
    
    for i in range(3, n-3):
        # Bullish OB: low of 3 bars before close breaks above recent high
        if (close[i+3] > np.max(high[i:i+3]) and 
            np.min(low[i:i+3]) == low[i]):  # lowest at start of base
            bullish_ob[i+3] = np.min(low[i:i+3])
        # Bearish OB: high of 3 bars before close breaks below recent low
        if (close[i+3] < np.min(low[i:i+3]) and 
            np.max(high[i:i+3]) == high[i]):  # highest at start of base
            bearish_ob[i+3] = np.max(high[i:i+3])
    
    # Forward fill to keep OB levels valid until broken
    bullish_ob_series = pd.Series(bullish_ob).ffill().values
    bearish_ob_series = pd.Series(bearish_ob).ffill().values
    
    # 60-period EMA for trend (6h timeframe)
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    uptrend_6h = close > ema_60
    downtrend_6h = close < ema_60
    
    # 12h trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.8 * 20-period average (avoid chop)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if OB levels not established
        if np.isnan(bullish_ob_series[i]) and np.isnan(bearish_ob_series[i]):
            signals[i] = 0.0
            continue
            
        bull_ob = bullish_ob_series[i]
        bear_ob = bearish_ob_series[i]
        uptrend = uptrend_6h[i]
        downtrend = downtrend_6h[i]
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price retraces to bullish OB in uptrend with volume
            if (not np.isnan(bull_ob) and 
                low[i] <= bull_ob * 1.005 and  # allow 0.5% slippage
                uptrend and uptrend_htf and vol_conf):
                signals[i] = 0.25
                position = 1
            # SHORT: price retraces to bearish OB in downtrend with volume
            elif (not np.isnan(bear_ob) and 
                  high[i] >= bear_ob * 0.995 and  # allow 0.5% slippage
                  downtrend and downtrend_htf and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches equilibrium (midpoint) or trend breaks
            eq_level = (bull_ob + close[i]) / 2 if not np.isnan(bull_ob) else close[i]
            if high[i] >= eq_level * 0.995 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches equilibrium or trend breaks
            eq_level = (bear_ob + close[i]) / 2 if not np.isnan(bear_ob) else close[i]
            if low[i] <= eq_level * 1.005 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals