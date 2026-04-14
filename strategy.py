#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) deviation with 1-day trend filter
# Long when price deviates below VWAP by 1.5 ATR AND 1-day EMA50 is rising
# Short when price deviates above VWAP by 1.5 ATR AND 1-day EMA50 is falling
# Exit when price returns to VWAP (mean reversion)
# VWAP captures institutional interest, deviation identifies overextension, daily EMA filters trend
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing mean reversion

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP on 4h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate ATR(14) for deviation threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA50 slope (rising/falling)
    ema50_slope = np.zeros_like(ema50_1d_aligned)
    ema50_slope[1:] = ema50_1d_aligned[1:] - ema50_1d_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        atr_val = atr[i]
        ema50_val = ema50_1d_aligned[i]
        slope = ema50_slope[i]
        
        if position == 0:
            # Long: price below VWAP by 1.5*ATR AND daily EMA50 rising
            if (price < vwap_val - 1.5 * atr_val) and (slope > 0):
                position = 1
                signals[i] = position_size
            # Short: price above VWAP by 1.5*ATR AND daily EMA50 falling
            elif (price > vwap_val + 1.5 * atr_val) and (slope < 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP (mean reversion)
            if price >= vwap_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP (mean reversion)
            if price <= vwap_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_Deviation_1dEMA50"
timeframe = "4h"
leverage = 1.0