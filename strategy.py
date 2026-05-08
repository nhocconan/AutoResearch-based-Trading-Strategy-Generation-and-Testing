#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h trend filter and 1d volume regime filter
# Long when price closes above 4h VWAP with 1d volume expansion, short when below with contraction
# Uses 4h for directional bias (trend), 1d for volume regime (high/low volatility states), 1h for precise entry/exit
# VWAP acts as dynamic support/resistance; volume regime filters for institutional participation
# Targets 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost
# Session filter (08-20 UTC) reduces noise from low-liquidity periods

name = "1h_VWAP_Trend_VolumeRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once for trend filter (VWAP)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h VWAP: typical price * volume / cumulative volume
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_array = vwap_4h.values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_array)
    
    # Get 1d data once for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume regime: current volume vs 20-day average (high/low volatility)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d  # >1 = high vol regime, <1 = low vol regime
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if outside session or critical data is NaN
        if not in_session[i] or \
           np.isnan(vwap_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vwap_4h_val = vwap_4h_aligned[i]
        vol_regime = vol_ratio_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Enter long: price above 4h VWAP in high volume regime (institutional buying)
            if price > vwap_4h_val and vol_regime > 1.2:
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h VWAP in low volume regime (lack of support)
            elif price < vwap_4h_val and vol_regime < 0.8:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h VWAP or volume regime shifts to low
            if price < vwap_4h_val or vol_regime < 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h VWAP or volume regime shifts to high
            if price > vwap_4h_val or vol_regime > 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals