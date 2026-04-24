#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme + 1d ATR-based Trend Filter + Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR-based trend filter (price > close + ATR(14) = uptrend, price < close - ATR(14) = downtrend).
- Entry: Long when Williams %R(14) crosses above -20 (exiting oversold) AND uptrend AND volume > 2.0 * 12h volume MA(20);
         Short when Williams %R(14) crosses below -80 (exiting overbought) AND downtrend AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when Williams %R crosses below -80; Short exits when Williams %R crosses above -20.
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R captures momentum exhaustion/reversal; ATR trend filter adapts to volatility; volume spike confirms conviction.
- Works in bull (buying oversold bounces in uptrend) and bear (selling overbought rejections in downtrend) with reduced whipsaws.
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
    
    # Get 1d data for ATR-based trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR-based trend: uptrend if price > previous close + ATR, downtrend if price < previous close - ATR
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first period
    trend_up = close_1d > prev_close_1d + atr_1d
    trend_down = close_1d < prev_close_1d - atr_1d
    
    # Align trend to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Williams %R on 12h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 12h data for volume MA(20)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 14)  # Williams %R needs 14, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter from 1d ATR-based trend
        uptrend = trend_up_aligned[i] == 1.0
        downtrend = trend_down_aligned[i] == 1.0
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_12h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: Williams %R crosses above -20 (exiting oversold)
                if prev_williams_r <= -20 and williams_r[i] > -20:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: Williams %R crosses below -80 (exiting overbought)
                if prev_williams_r >= -80 and williams_r[i] < -80:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -80
            if prev_williams_r > -80 and williams_r[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -20
            if prev_williams_r < -20 and williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dATRTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0