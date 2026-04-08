#!/usr/bin/env python3
# 12h_camarilla_volume_trend_v1
# Hypothesis: On 12h timeframe, trade Camarilla pivot breakouts with volume confirmation and daily trend filter.
# Uses 1d Camarilla levels (H4/L4) for entry, 1d volume spike for confirmation, and 1d EMA50 for trend filter.
# Works in bull/bear by following higher timeframe trend. Low trade frequency (~15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla, volume, and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H4, L4) from previous day
    # H4 = Close + 1.1/2 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1/2 * range_1d
    camarilla_l4 = close_1d - 1.1/2 * range_1d
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily volume average (20-period) for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below L4 OR volume drops below average OR trend turns bearish
            if (close[i] < camarilla_l4_aligned[i]) or (volume[i] < vol_ma_20_aligned[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above H4 OR volume drops below average OR trend turns bullish
            if (close[i] > camarilla_h4_aligned[i]) or (volume[i] < vol_ma_20_aligned[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
            
            # Long entry: price breaks above H4 with volume confirmation and uptrend
            if (close[i] > camarilla_h4_aligned[i]) and volume_confirm and (close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L4 with volume confirmation and downtrend
            elif (close[i] < camarilla_l4_aligned[i]) and volume_confirm and (close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals