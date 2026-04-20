#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-Weighted Average Price (VWAP) deviation with Daily Trend Filter
# - Price deviation from 12h VWAP for mean reversion signals
# - Daily EMA(50) as trend filter: long when price > EMA50, short when price < EMA50
# - Only take trades when price deviates >1.5% from VWAP and aligns with daily trend
# - VWAP provides dynamic support/resistance, daily trend filters counter-trend moves
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h VWAP
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate price deviation from VWAP (%)
    price = prices['close'].values
    vwap_dev = (price - vwap.values) / vwap.values * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema_50_aligned[i]) or np.isnan(vwap_dev[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine price position relative to daily EMA50
        price_above_ema = price[i] > ema_50_aligned[i]
        price_below_ema = price[i] < ema_50_aligned[i]
        
        # VWAP deviation thresholds
        vwap_dev_long = vwap_dev[i] < -1.5  # Price >1.5% below VWAP -> long signal
        vwap_dev_short = vwap_dev[i] > 1.5   # Price >1.5% above VWAP -> short signal
        
        if position == 0:
            # Long entry: price below VWAP + above daily EMA50 (uptrend pullback)
            if vwap_dev_long and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price above VWAP + below daily EMA50 (downtrend bounce)
            elif vwap_dev_short and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses above VWAP or below daily EMA50
            if vwap_dev[i] > 0 or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below VWAP or above daily EMA50
            if vwap_dev[i] < 0 or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VWAP_Deviation_DailyEMAFilter"
timeframe = "12h"
leverage = 1.0