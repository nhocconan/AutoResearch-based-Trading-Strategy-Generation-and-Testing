#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 Breakout + 1d EMA34 Trend Filter + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Camarilla R1 level AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla S1 level AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when price closes below Camarilla Pivot point; Short exits when price closes above Camarilla Pivot point.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels from 1d provide precise intraday support/resistance; EMA34 filters higher-timeframe trend; volume spike confirms conviction.
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
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = high - low
    # R4 = close + Range * 1.1/2, R3 = close + Range * 1.1/4, R2 = close + Range * 1.1/6, R1 = close + Range * 1.1/12
    # S1 = close - Range * 1.1/12, S2 = close - Range * 1.1/6, S3 = close - Range * 1.1/4, S4 = close - Range * 1.1/2
    # Pivot = (high + low + close) / 3
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + rng * 1.1 / 12
    camarilla_s1 = close_1d - rng * 1.1 / 12
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_open = open_[i]
        curr_volume = volume[i]
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Camarilla R1
                if curr_open <= camarilla_r1_aligned[i] and curr_close > camarilla_r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Camarilla S1
                if curr_open >= camarilla_s1_aligned[i] and curr_close < camarilla_s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price closes below Camarilla Pivot
            if curr_close < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla Pivot
            if curr_close > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0