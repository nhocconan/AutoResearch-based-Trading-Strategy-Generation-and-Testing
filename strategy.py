#!/usr/bin/env python3
"""
4h_ThreeStooges_LongOnly_V1
Long-only strategy for BTC/ETH using daily VWAP trend, hourly VWAP deviation, and volume confirmation.
- Long when: price > daily VWAP (trend), price < 0.5 std below hourly VWAP (dip), and volume > 1.5x 20-period average
- Exit when: price crosses above hourly VWAP (mean reversion complete) or trend breaks
- Designed for ~25-35 trades/year per symbol (100-140 total over 4 years)
- Works in bull markets (buy dips in uptrend) and bear markets (buy dips in less steep downtrends)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP trend
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Get 1h data for VWAP and bands
    df_1h = get_htf_data(prices, '1h')
    typical_price_1h = (df_1h['high'] + df_1h['low'] + df_1h['close']) / 3.0
    vwap_1h = (typical_price_1h * df_1h['volume']).cumsum() / df_1h['volume'].cumsum()
    vwap_1h = vwap_1h.values
    
    # Calculate rolling std of deviation from VWAP (1h)
    deviation = typical_price_1h - vwap_1h
    # Use expanding std for first 20 periods, then rolling
    deviation_std = pd.Series(deviation).expanding().std().values
    deviation_std_20 = pd.Series(deviation).rolling(window=20, min_periods=20).std().values
    # Combine: use expanding until we have 20 samples, then rolling
    vwap_std = np.where(np.arange(len(deviation)) < 20, deviation_std, deviation_std_20)
    
    # Align all data to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_1h_aligned = align_htf_to_ltf(prices, df_1h, vwap_1h)
    vwap_std_aligned = align_htf_to_ltf(prices, df_1h, vwap_std)
    
    # Volume average (20-period) on 1h
    vol_ma_20_1h = pd.Series(df_1h['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 20  # need enough for VWAP std and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_1h_aligned[i]) or 
            np.isnan(vwap_std_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: price > daily VWAP
        uptrend = close[i] > vwap_1d_aligned[i]
        
        # Dip condition: price < 0.5 std below 1h VWAP
        dip_condition = close[i] < (vwap_1h_aligned[i] - 0.5 * vwap_std_aligned[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Exit condition: price crosses above 1h VWAP (mean reversion)
        exit_condition = close[i] > vwap_1h_aligned[i]
        
        if position == 0:
            # Long: uptrend + dip + volume
            if uptrend and dip_condition and vol_confirm:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: mean reversion complete or trend break
            if exit_condition or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_ThreeStooges_LongOnly_V1"
timeframe = "4h"
leverage = 1.0