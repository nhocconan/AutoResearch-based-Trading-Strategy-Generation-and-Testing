#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1-day volume-weighted VWAP deviation
# Choppiness Index identifies ranging (chop>61.8) vs trending (chop<38.2) markets
# In ranging markets: mean reversion at VWAP deviation > 1.5 std
# In trending markets: momentum continuation when price > VWAP + 0.5*std (long) or < VWAP - 0.5*std (short)
# Volume-weighted VWAP acts as dynamic support/resistance
# Designed for 4h timeframe with selective entries to avoid overtrading
# Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data for VWAP and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day VWAP (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    
    # Calculate 1-day true range for Choppiness Index
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Calculate 14-period ATR and highest/lowest for Choppiness Index
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh14_ll14_diff = highest_high_1d - lowest_low_1d
    # Avoid division by zero
    chop_raw = np.where(hh14_ll14_diff > 0, sum_atr_14 / hh14_ll14_diff, 1.0)
    chop_1d = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align 1-day indicators to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h VWAP
    typical_price = (high + low + close) / 3.0
    vwap_num_4h = np.cumsum(typical_price * volume)
    vwap_den_4h = np.cumsum(volume)
    vwap_4h = vwap_num_4h / vwap_den_4h
    
    # Calculate 4h ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period standard deviation of price deviation from VWAP
    price_dev = close - vwap_4h
    dev_ma = pd.Series(price_dev).rolling(window=20, min_periods=20).mean().values
    dev_std = pd.Series(price_dev - dev_ma).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or \
           np.isnan(vwap_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(dev_std[i]) or dev_std[i] == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime classification based on 1-day Choppiness Index
        is_ranging = chop_1d_aligned[i] > 61.8
        is_trending = chop_1d_aligned[i] < 38.2
        
        # VWAP deviation measures
        dev_from_vwap = close[i] - vwap_4h[i]
        dev_normalized = dev_from_vwap / dev_std[i] if dev_std[i] > 0 else 0
        
        price = close[i]
        
        if position == 0:
            if is_ranging:
                # Mean reversion in ranging markets
                long_signal = dev_normalized < -1.5  # Price significantly below VWAP
                short_signal = dev_normalized > 1.5   # Price significantly above VWAP
            elif is_trending:
                # Momentum continuation in trending markets
                long_signal = dev_normalized > 0.5   # Price above VWAP + 0.5*std
                short_signal = dev_normalized < -0.5  # Price below VWAP - 0.5*std
            else:
                # Neutral chop - no clear signal
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or mean reversion signal
            stop_loss = entry_price - 2.0 * atr_4h[i]
            mean_revert_signal = dev_normalized > 0.2  # Return toward VWAP
            
            if stop_loss <= 0 or price <= stop_loss or mean_revert_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or mean reversion signal
            stop_loss = entry_price + 2.0 * atr_4h[i]
            mean_revert_signal = dev_normalized < -0.2  # Return toward VWAP
            
            if stop_loss <= 0 or price >= stop_loss or mean_revert_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChopRegime_VWAPDeviation"
timeframe = "4h"
leverage = 1.0