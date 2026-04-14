#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ATR volatility filter
# Donchian breakout captures momentum in trending markets
# Volume > 1.5x average confirms institutional participation
# ATR(1w) filter: trade only when ATR(1w) > median ATR(1w) to avoid low volatility periods
# Works in bull/bear as breakouts occur in both regimes with volume confirmation
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volume and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE for ATR filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d volume moving average (20 periods)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate 1w ATR (14 periods) and its median for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median = np.nanmedian(atr_1w[~np.isnan(atr_1w)])
    atr_filter = atr_1w > atr_median
    atr_filter_aligned = align_htf_to_ltf(prices, df_1w, atr_filter, additional_delay_bars=0)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR(1w) > median
        if not atr_filter_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + volume confirmation
            if close[i] > dc_upper[i] and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + volume confirmation
            elif close[i] < dc_lower[i] and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian lower band
            if close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian upper band
            if close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Volume_1wATR_Filter_Donchian_Breakout_v1"
timeframe = "4h"
leverage = 1.0