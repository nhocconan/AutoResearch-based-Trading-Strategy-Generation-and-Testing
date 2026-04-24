#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 Breakout with 1d EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Camarilla H4 level AND price > 1d EMA50 AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Camarilla L4 level AND price < 1d EMA50 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when price crosses below Camarilla L4 level; Short exits when price crosses above Camarilla H4 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla H4/L4 levels use the classic formula: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2.
- Uses 1d Camarilla levels applied to 12h chart with proper MTF alignment.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
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
    
    # Get 1d data for Camarilla levels (H4, L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for meaningful calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # H4 = close + 1.1*(high - low)/2
    # L4 = close - 1.1*(high - low)/2
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 1d data for EMA50 trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 12h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
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
                # Long: price breaks above Camarilla H4 level AND price > 1d EMA50 (uptrend)
                if curr_close > camarilla_h4_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Camarilla L4 level AND price < 1d EMA50 (downtrend)
                elif curr_close < camarilla_l4_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla L4 level
            if curr_close < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Camarilla H4 level
            if curr_close > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0