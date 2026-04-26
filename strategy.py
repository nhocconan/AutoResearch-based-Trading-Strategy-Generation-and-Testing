#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: 4h Camarilla R3/S3 breakout with daily EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 level AND daily EMA34 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Camarilla S3 level AND daily EMA34 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Camarilla pivot levels from 1d chart for structure-based breakouts
- Daily EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 19-50 trades/year on 4h) to minimize fee drag
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Combines proven Camarilla breakout with EMA trend and volume confirmation - different from saturated variants
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
    
    # Load daily data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels on 4h chart (primary timeframe) using previous 1d OHLC
    # We need to get the previous day's OHLC for each 4h bar
    # Since we're on 4h timeframe, we can use the 1d data to calculate pivots for the current day
    # For each 4h bar, we use the previous completed day's OHLC
    
    # Get previous day's OHLC (aligned to 4h chart)
    # We'll shift the 1d data by 1 to get previous day's values
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_open_1d = df_1d['open'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # Camarilla formulas:
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # R2 = close + ((high-low) * 1.1/6)
    # R1 = close + ((high-low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high-low) * 1.1/12)
    # S2 = close - ((high-low) * 1.1/6)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    
    # We only need R3 and S3 for breakout
    prev_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + (prev_range * 1.1 / 4)
    camarilla_s3 = prev_close_1d - (prev_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h chart (they stay constant throughout the day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for daily EMA, 20 for volume MA)
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
            # Long: Price breaks above Camarilla R3 AND daily uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND daily downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR daily trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR daily trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0