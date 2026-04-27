#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe act as strong support/resistance.
Breakouts with volume confirmation and 1d trend filter capture institutional moves.
Designed for low trade frequency (target 12-37/year) on 12h timeframe to minimize fee drag.
Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Range = high - low
    daily_range = high - low
    
    # Camarilla levels (based on previous day)
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # R2 = close + range * 1.1/6
    # R1 = close + range * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.1/12
    # S2 = close - range * 1.1/6
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    
    # We need previous day's data for today's levels
    # Shift by 1 to use previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan  # First bar has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    prev_typical = (prev_high + prev_low + prev_close) / 3
    prev_range = prev_high - prev_low
    
    # Calculate Camarilla levels
    R3 = prev_close + prev_range * 1.1 / 4
    S3 = prev_close - prev_range * 1.1 / 4
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 2.0 * 24-period average (2 days of 12h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align 1d indicators to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1 period for Camarilla (uses prev day), 34 for EMA, 24 for volume avg
    start_idx = max(1, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_level = R3[i]
        s3_level = S3[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and vol_conf:
                # Long: break above R3 with volume
                if close_val > r3_level:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S3 with volume
                if close_val < s3_level:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below R3 or trend reversal
            if close_val < r3_level:  # Re-enter below R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above S3 or trend reversal
            if close_val > s3_level:  # Re-enter above S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0