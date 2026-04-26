#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 level AND 4h EMA50 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Camarilla S3 level AND 4h EMA50 downtrend AND volume > 1.5 * volume_ma(20)
- Uses 1h for entry timing precision, 4h for signal direction to minimize fee drag
- Volume spike ensures institutional participation and reduces false breakouts
- Target: 15-35 trades/year (60-140 over 4 years) to stay within fee drag limits
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Designed to work in both bull (breakouts with trend) and bear (failed breaks as continuation) markets
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
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter (needs completed 4h candle)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_4h = np.where(ema_50_4h_aligned > 0, 
                        np.where(close > ema_50_4h_aligned, 1, -1), 
                        0)
    
    # Calculate volume spike filter: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Calculate Camarilla levels from prior 1h bar (completed bar only)
    # Camarilla levels based on previous bar's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Using prior completed 1h bar to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20) + 1  # +1 for prior bar
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trend_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 4h uptrend AND volume spike
            if close[i] > camarilla_r3[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 AND 4h downtrend AND volume spike
            elif close[i] < camarilla_s3[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S3 OR 4h trend turns down
            if close[i] < camarilla_s3[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R3 OR 4h trend turns up
            if close[i] > camarilla_r3[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0