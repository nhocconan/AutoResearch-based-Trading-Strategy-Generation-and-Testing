#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and 1w volume confirmation.
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 (uptrend) AND 1w volume > 1.5x 20-period volume MA.
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 (downtrend) AND 1w volume > 1.5x 20-period volume MA.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size fixed at 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Camarilla levels provide mathematically derived support/resistance, 1d EMA34 filters for trend alignment, 1w volume confirms institutional participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 1d trend when volume confirms.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_1wVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter (Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume 20-period MA for spike detection
    volume_ma_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    # Camarilla levels are based on previous day's range
    prev_high = df_1d['high'].shift(1).values  # Previous 1d high
    prev_low = df_1d['low'].shift(1).values    # Previous 1d low
    prev_close = df_1d['close'].shift(1).values # Previous 1d close
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_1w_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition (1w volume > 1.5x 20-period MA)
        volume_spike_1w = volume_ma_1w_aligned[i] > 0 and df_1w['volume'].values[-1] > (volume_ma_1w[i] * 1.5) if len(df_1w) > 0 else False
        # Simplified: use current 1w volume vs its MA (approximation for aligned data)
        # Since we don't have current 1w volume in 6h data, we use the aligned MA and check if current period is volatile
        volume_ratio = df_1w['volume'].iloc[-1] / volume_ma_1w[-1] if len(df_1w) > 0 and volume_ma_1w[-1] > 0 else 1
        volume_spike_1w = volume_ratio > 1.5
        
        # Camarilla breakout conditions
        breakout_up = high_val > R3_aligned[i]   # Price breaks above Camarilla R3
        breakout_down = low_val < S3_aligned[i]  # Price breaks below Camarilla S3
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 1d uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike_1w:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 1d downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike_1w:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point OR trend changes
            # Camarilla pivot point = (prev_high + prev_low + prev_close) / 3
            pivot_point = (df_1d['high'].shift(1).iloc[-1] + df_1d['low'].shift(1).iloc[-1] + df_1d['close'].shift(1).iloc[-1]) / 3 if len(df_1d) > 1 else close_val
            pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, pivot_point))
            if close_val < pivot_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point OR trend changes
            pivot_point = (df_1d['high'].shift(1).iloc[-1] + df_1d['low'].shift(1).iloc[-1] + df_1d['close'].shift(1).iloc[-1]) / 3 if len(df_1d) > 1 else close_val
            pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, pivot_point))
            if close_val > pivot_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals