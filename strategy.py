#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout with daily EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND daily EMA34 uptrend AND volume > 1.5x volume MA(20)
- Short when price breaks below Camarilla S1 AND daily EMA34 downtrend AND volume > 1.5x volume MA(20)
- Uses Camarilla pivot levels from prior completed 1d bar (no look-ahead)
- Daily EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike filter confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 19-50 trades/year) to minimize fee drag
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal
- Works in both bull and bear markets by following daily trend and requiring volume confirmation
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
    
    # Load daily data ONCE before loop for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d bar (completed bar only)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    camarilla_high = typical_price + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_low = typical_price - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (available after 1d bar closes)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low.values)
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Volume spike filter: volume > 1.5x volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND daily uptrend AND volume spike
            if close[i] > camarilla_high_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND daily downtrend AND volume spike
            elif close[i] < camarilla_low_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR daily trend turns down
            if close[i] < camarilla_low_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR daily trend turns up
            if close[i] > camarilla_high_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0