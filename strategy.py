#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeConfirm_v2
Hypothesis: 12h Camarilla pivot breakout strategy with 1d trend filter and volume confirmation.
- Uses 12h timeframe for low trade frequency (target: 50-150 total trades over 4 years)
- Camarilla levels (R3, S3) calculated from prior 1d candle
- Long when price breaks above R3 with volume > 1.5x 20-period average AND 1d close > 1d EMA34
- Short when price breaks below S3 with volume > 1.5x 20-period average AND 1d close < 1d EMA34
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by aligning with 1d trend and using Camarilla breakouts for entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from prior daily OHLC
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    camarilla_R3 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    camarilla_S3 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA, and 1d data alignment)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        price_above_R3 = close[i] > camarilla_R3_aligned[i]
        price_below_S3 = close[i] < camarilla_S3_aligned[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema34_1d_aligned[i]
        daily_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 AND volume confirmation AND daily uptrend
            if price_above_R3 and volume_confirm[i] and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume confirmation AND daily downtrend
            elif price_below_S3 and volume_confirm[i] and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 (reversal) OR daily trend turns down
            if price_below_S3 or not daily_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 (reversal) OR daily trend turns up
            if price_above_R3 or not daily_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeConfirm_v2"
timeframe = "12h"
leverage = 1.0