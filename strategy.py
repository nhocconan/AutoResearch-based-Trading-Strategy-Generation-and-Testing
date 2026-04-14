#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Volume Weighted Average Price (VWAP) with 1w ATR-based volatility filter and price action confirmation.
# Long when price closes above 1d VWAP and closes above prior 12h high, with 1w ATR(14) > 30-period SMA(ATR) indicating expanding volatility.
# Short when price closes below 1d VWAP and closes below prior 12h low, with same volatility expansion condition.
# Exit when price crosses back below/above 1d VWAP or volatility contracts (ATR < SMA(ATR)).
# Designed to capture institutional breakouts with volume confirmation in both bull and bear markets.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for VWAP and ATR calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap = vwap_numerator / vwap_denominator
    # Handle division by zero at start
    vwap = np.where(vwap_denominator == 0, typical_price, vwap)
    
    # Calculate 1w ATR for volatility filter (need to load 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate ATR SMA for volatility expansion filter
    atr_sma = pd.Series(atr_1w_aligned).rolling(window=30, min_periods=30).mean().values
    
    # Calculate 12h high/low for price action confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Align 12h high/low to 12h timeframe (same timeframe, no alignment needed but using same method)
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need VWAP and ATR periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(atr_sma[i]) or
            np.isnan(high_12h_aligned[i]) or
            np.isnan(low_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: expanding volatility (ATR > SMA of ATR)
        volatility_expanding = atr_1w_aligned[i] > atr_sma[i]
        
        # Price action: close above/below prior 12h high/low
        price_above_prior_high = close[i] > high_12h_aligned[i-1] if i > 0 else False
        price_below_prior_low = close[i] < low_12h_aligned[i-1] if i > 0 else False
        
        # VWAP relationship
        price_above_vwap = close[i] > vwap_aligned[i]
        price_below_vwap = close[i] < vwap_aligned[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and volatility expansion
            # Long: price closes above VWAP AND above prior 12h high AND volatility expanding
            if (price_above_vwap and 
                price_above_prior_high and 
                volatility_expanding):
                position = 1
                signals[i] = position_size
            # Short: price closes below VWAP AND below prior 12h low AND volatility expanding
            elif (price_below_vwap and 
                  price_below_prior_low and 
                  volatility_expanding):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or volatility contracts
            if (price_below_vwap or 
                not volatility_expanding):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above VWAP or volatility contracts
            if (price_above_vwap or 
                not volatility_expanding):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_VWAP_1w_ATR_VolatilityFilter_v1"
timeframe = "12h"
leverage = 1.0