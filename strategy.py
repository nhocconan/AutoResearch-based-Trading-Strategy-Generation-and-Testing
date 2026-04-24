#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot Breakout + 1w EMA50 Trend + Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Camarilla R3 level AND price > 1w EMA50 AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Camarilla S3 level AND price < 1w EMA50 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when price crosses below Camarilla H3 level; Short exits when price crosses above Camarilla L3 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla pivot levels provide intraday support/resistance; 1w EMA50 filters higher-timeframe trend; volume spike confirms conviction.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least previous day
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 12h bar using previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's range)
    # R4 = close + ((high - low) * 1.5000)
    # R3 = close + ((high - low) * 1.2500)
    # R2 = close + ((high - low) * 1.1666)
    # R1 = close + ((high - low) * 1.0833)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.0833)
    # S2 = close - ((high - low) * 1.1666)
    # S3 = close - ((high - low) * 1.2500)
    # S4 = close - ((high - low) * 1.5000)
    
    # We need to align these to 12h timeframe
    # For each 12h bar, we use the previous completed 1d bar
    range_1d = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + (range_1d * 1.2500)
    camarilla_s3 = close_1d - (range_1d * 1.2500)
    camarilla_h3 = camarilla_pp + (range_1d * 1.0833)  # H3 is same as R1
    camarilla_l3 = camarilla_pp - (range_1d * 1.0833)  # L3 is same as S1
    
    # Align Camarilla levels to 12h timeframe (using previous day's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 12h data for volume MA(20)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Camarilla R3
                if curr_close > camarilla_r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Camarilla S3
                if curr_close < camarilla_s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla H3
            if curr_close < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Camarilla L3
            if curr_close > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0