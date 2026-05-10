#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Camarilla pivot levels (R1, S1) from daily timeframe act as intraday support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily EMA34 trend filter capture
# institutional moves. Works in bull/bear markets by trading breakouts in direction of higher timeframe trend.
# Uses volume spike (2.0x average) to filter weak breakouts and avoid false signals.
# Target: 25-35 trades/year to minimize fee drag while capturing significant moves.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
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
    
    # Get 1-day data for Camarilla calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for the previous day (to avoid look-ahead)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C, H, L are from previous day
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day will have NaN due to roll, handled later
    
    cam_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + cam_range * 1.1 / 12
    s1 = prev_close_1d - cam_range * 1.1 / 12
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average (strong breakout filter)
        volume_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: breakout above R1 with volume confirmation and price above daily EMA34 (uptrend bias)
            if close[i] > r1_aligned[i] and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: breakdown below S1 with volume confirmation and price below daily EMA34 (downtrend bias)
            elif close[i] < s1_aligned[i] and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal) or loses trend
            if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal) or loses trend
            if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals