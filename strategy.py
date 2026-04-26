#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of 12h trend to avoid counter-trend whipsaws. Uses volume spike (2.0x 20-bar average) to confirm momentum.
Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year) by requiring confluence of breakout, trend, and volume.
Designed for BTC/ETH - Camarilla pivots work in ranging markets, trend filter avoids false breakouts in strong trends, volume confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for HTF trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    htf_trend = np.where(close > ema_50_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla levels from previous 1d bar (standard formula)
    # Camarilla levels use previous day's OHLC
    df_1d = get_htf_data(prices, '1d')
    # Previous day's close, high, low (we need the completed 1d bar)
    prev_close = df_1d['close'].shift(1).values  # Shift by 1 to get previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for Camarilla shift)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Price relative to Camarilla levels
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        price_above_R2 = close[i] > R2_aligned[i]
        price_below_S2 = close[i] < S2_aligned[i]
        
        if htf_trend[i] == 1:  # Uptrend on 12h
            # Long signal: price breaks above R1 with volume spike
            if price_above_R1 and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: price falls below S1
            elif price_below_S1:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 12h
            # Short signal: price breaks below S1 with volume spike
            if price_below_S1 and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: price rises above R1
            elif price_above_R1:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0