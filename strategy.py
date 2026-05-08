#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume Weighted Average Price (VWAP) deviation with 1d trend filter
# Long when price crosses above VWAP in uptrend, short when crosses below VWAP in downtrend
# Uses volume confirmation and avoids overtrading by requiring strong deviations
# Designed to work in both bull and bear markets by following higher timeframe trend
# Target: 25-75 total trades over 4 years (6-19/year)

name = "4h_VWAP_Deviation_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # VWAP deviation bands (1.5 * ATR)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    upper_band = vwap + (1.5 * atr)
    lower_band = vwap - (1.5 * atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        vwap_val = vwap[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        if position == 0:
            # Enter long: price crosses above VWAP in uptrend
            if price > vwap_val and price > upper and close[i-1] <= upper_band[i-1] and ema34_1d_val > vwap_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP in downtrend
            elif price < vwap_val and price < lower and close[i-1] >= lower_band[i-1] and ema34_1d_val < vwap_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below VWAP or trend changes
            if price < vwap_val or ema34_1d_val < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above VWAP or trend changes
            if price > vwap_val or ema34_1d_val > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals