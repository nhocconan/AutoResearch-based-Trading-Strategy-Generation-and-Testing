# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 breakouts with 1d trend filter and volume spikes capture momentum.
# Works in bull via R3 breakouts, bear via S3 breakdowns. Volume filter reduces false signals.
# Target: 20-50 trades/year with position size 0.25.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # For each 4h bar, use previous day's OHLC
    prev_day_high = pd.Series(high).rolling(window=24, min_periods=1).max().shift(24).values
    prev_day_low = pd.Series(low).rolling(window=24, min_periods=1).min().shift(24).values
    prev_day_close = pd.Series(close).rolling(window=24, min_periods=1).mean().shift(24).values
    
    # Calculate Camarilla R3 and S3 levels
    R3 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 4
    S3 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 4
    
    # Volume ratio: current volume / 24-period average volume (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need previous day data
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(R3[i]) or np.isnan(S3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above R3 or below S3
        breakout_up = close[i] > R3[i]
        breakout_down = close[i] < S3[i]
        
        # Volume confirmation: volume > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below S3 or trend reversal
            if close[i] < S3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above R3 or trend reversal
            if close[i] > R3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals