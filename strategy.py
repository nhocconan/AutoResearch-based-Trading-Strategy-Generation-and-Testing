#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Camarilla H3 level AND price > 1d EMA34 AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below Camarilla L3 level AND price < 1d EMA34 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when price crosses below Camarilla L3 level; Short exits when price crosses above Camarilla H3 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
- Uses 1d Camarilla levels applied to 6h chart with proper MTF alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for meaningful calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1d data for EMA34 trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Camarilla H3 level AND price > 1d EMA34 (uptrend)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Camarilla L3 level AND price < 1d EMA34 (downtrend)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla L3 level
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Camarilla H3 level
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0