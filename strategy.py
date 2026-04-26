#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Camarilla pivot levels from prior 1d bar (completed bar only)
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag on 6h timeframe
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Combines Camarilla breakouts with HTF trend and volume confirmation for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Camarilla levels (structure)
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate prior 6h bar's OHLC for Camarilla pivot
    # We need the completed 6h bar's OHLC, so we use shift(1) on the 6h data
    # But since we're using get_htf_data which gives us completed bars, we can use the last completed bar
    # For each 6h bar, we use the previous completed 6h bar's OHLC
    h_6h = df_6h['high'].values
    l_6h = df_6h['low'].values
    c_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for each 6h bar using prior 6h bar's OHLC
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We need to shift the OHLC by 1 to get the prior bar's values
    h_prev = np.roll(h_6h, 1)
    l_prev = np.roll(l_6h, 1)
    c_prev = np.roll(c_6h, 1)
    # Set first value to NaN since there's no prior bar
    h_prev[0] = np.nan
    l_prev[0] = np.nan
    c_prev[0] = np.nan
    
    camarilla_r3 = c_prev + (h_prev - l_prev) * 1.1 / 4
    camarilla_s3 = c_prev - (h_prev - l_prev) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1d trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1d trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0