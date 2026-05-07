#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h trend (price above/below EMA50) for direction and 1d volume spike for confirmation.
# Enter on 1h Camarilla R1/S1 breakout with volume > 1.5x 20-period average in the direction of 4h trend.
# Exit when price crosses back below/above the Camarilla pivot point or trend reverses.
# Position size: 0.20 to limit drawdown. Target: 15-35 trades/year (~60-140 over 4 years).

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d 20-period average volume for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels for 1h: based on previous bar's high, low, close
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous bar's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]
    plow[0] = low[0]
    pclose[0] = close[0]
    
    rang = phigh - plow
    r1 = pclose + rang * 1.1 / 12
    s1 = pclose - rang * 1.1 / 12
    pivot = (phigh + plow + pclose) / 3  # Standard pivot for exit
    
    # Volume ratio: current volume / 1d average volume
    vol_ratio = np.where(vol_ma_20_1d_aligned > 0, volume / vol_ma_20_1d_aligned, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pivot[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 4h EMA50
        uptrend_regime = close[i] > ema_50_4h_aligned[i]
        downtrend_regime = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above R1 in uptrend regime + volume confirmation
            long_entry = (close[i] > r1[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below S1 in downtrend regime + volume confirmation
            short_entry = (close[i] < s1[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or trend reverses to downtrend
            if (close[i] < pivot[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses above pivot or trend reverses to uptrend
            if (close[i] > pivot[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals