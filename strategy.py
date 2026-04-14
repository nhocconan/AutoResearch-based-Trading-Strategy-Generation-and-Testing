# 12h_1w_Camarilla_Breakout_Volume_Filter_v1
# Uses weekly Camarilla pivot levels from 1w for breakout signals
# 1d EMA (50) as trend filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 1d trend
# Target: 15-40 trades/year (60-160 total over 4 years) to minimize fee drag

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla levels
    range_1w = high_1w - low_1w
    camarilla_h4 = close_1w + range_1w * 1.500
    camarilla_h3 = close_1w + range_1w * 1.250
    camarilla_h2 = close_1w + range_1w * 1.166
    camarilla_h1 = close_1w + range_1w * 1.083
    camarilla_l1 = close_1w - range_1w * 1.083
    camarilla_l2 = close_1w - range_1w * 1.166
    camarilla_l3 = close_1w - range_1w * 1.250
    camarilla_l4 = close_1w - range_1w * 1.500
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x average volume (30-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=30, min_periods=30).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for EMA and Camarilla calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above H4 with volume filter and above 1d EMA
            if price > camarilla_h4_aligned[i] and vol > 1.5 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below L4 with volume filter and below 1d EMA
            elif price < camarilla_l4_aligned[i] and vol > 1.5 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L4 (reversal) or below 1d EMA
            if price < camarilla_l4_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H4 (reversal) or above 1d EMA
            if price > camarilla_h4_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Camarilla_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0