#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d VWAP deviation and 1w ATR-based volatility filter
# VWAP deviation identifies mean reversion opportunities when price deviates significantly from VWAP
# ATR volatility filter ensures trades occur in sufficient volatility environments
# Works in both bull and bear markets as it captures mean reversion during extended moves
# Uses 1d VWAP for mean reversion signal and 1w ATR for volatility regime filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d VWAP (Volume Weighted Average Price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    vwap = vwap.replace([np.inf, -np.inf], np.nan).ffill().values
    
    # Calculate price deviation from VWAP (normalized by price)
    price_dev = (df_1d['close'] - vwap) / vwap
    price_dev = np.where(vwap == 0, 0, price_dev)
    
    # Align price deviation to 12h timeframe
    price_dev_aligned = align_htf_to_ltf(prices, df_1d, price_dev)
    
    # Load 1w data ONCE for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ATR (14 periods) for volatility filter
    atr_length = 14
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_length, adjust=False, min_periods=atr_length).mean().values
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, atr_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(price_dev_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR above 20th percentile of last 20 periods
        if i >= 20:
            atr_slice = atr_aligned[max(0, i-20):i]
            if len(atr_slice) > 0:
                # Calculate percentile of current ATR vs last 20 values
                valid_atr = atr_slice[~np.isnan(atr_slice)]
                if len(valid_atr) > 0:
                    sorted_atr = np.sort(valid_atr)
                    current_atr = atr_aligned[i]
                    rank = np.searchsorted(sorted_atr, current_atr, side='left')
                    percentile = (rank / len(sorted_atr)) * 100
                    sufficient_vol = percentile >= 20  # Sufficient volatility
                else:
                    sufficient_vol = False
            else:
                sufficient_vol = False
        else:
            sufficient_vol = False
        
        if position == 0:
            # Enter long: price significantly below VWAP + sufficient volatility
            if price_dev_aligned[i] < -0.02 and sufficient_vol:  # 2% below VWAP
                position = 1
                signals[i] = position_size
            # Enter short: price significantly above VWAP + sufficient volatility
            elif price_dev_aligned[i] > 0.02 and sufficient_vol:  # 2% above VWAP
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or volatility drops
            if price_dev_aligned[i] >= -0.005 or not sufficient_vol:  # Within 0.5% of VWAP or low vol
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or volatility drops
            if price_dev_aligned[i] <= 0.005 or not sufficient_vol:  # Within 0.5% of VWAP or low vol
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dVWAP_Dev_1wATR_VolFilter_v1"
timeframe = "12h"
leverage = 1.0