# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation.
- Uses 4h Supertrend for trend direction (avoids whipsaw in chop)
- Uses 1d RSI for regime filter (avoids counter-trend in strong trends)
- Uses 1h EMA crossover for entry timing (fast but filtered by HTF)
- Volume spike filter ensures momentum legitimacy
- Targets 15-30 trades/year by requiring confluence of multiple filters
- Works in bull (follows trend) and bear (avoids counter-trend via RSI filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Supertrend (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR for Supertrend (10-period)
    tr_4h = np.zeros(len(df_4h))
    tr_4h[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(df_4h)):
        tr_4h[i] = max(
            high_4h[i] - low_4h[i],
            abs(high_4h[i] - close_4h[i-1]),
            abs(low_4h[i] - close_4h[i-1])
        )
    
    atr_4h = np.full(len(df_4h), np.nan)
    if len(df_4h) >= 10:
        atr_4h[9] = np.mean(tr_4h[:10])
        for i in range(10, len(df_4h)):
            atr_4h[i] = (atr_4h[i-1] * 9 + tr_4h[i]) / 10
    
    # Calculate 4h Supertrend (10, 3.0)
    hl_avg_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl_avg_4h + 3.0 * atr_4h
    lower_band_4h = hl_avg_4h - 3.0 * atr_4h
    
    supertrend_4h = np.full(len(df_4h), np.nan)
    dir_4h = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if close_4h[i] > upper_band_4h[i-1]:
            dir_4h[i] = 1
        elif close_4h[i] < lower_band_4h[i-1]:
            dir_4h[i] = -1
        else:
            dir_4h[i] = dir_4h[i-1]
        
        if dir_4h[i] == 1:
            supertrend_4h[i] = max(lower_band_4h[i], supertrend_4h[i-1] if not np.isnan(supertrend_4h[i-1]) else lower_band_4h[i])
        else:
            supertrend_4h[i] = min(upper_band_4h[i], supertrend_4h[i-1] if not np.isnan(supertrend_4h[i-1]) else upper_band_4h[i])
    
    # Align Supertrend direction to 1h
    dir_4h_aligned = align_htf_to_ltf(prices, df_4h, dir_4h)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Load 1d data for RSI (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI (14-period)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    avg_gain_1d = np.full(len(df_1d), np.nan)
    avg_loss_1d = np.full(len(df_1d), np.nan)
    
    if len(df_1d) >= 14:
        avg_gain_1d[13] = np.mean(gain_1d[:14])
        avg_loss_1d[13] = np.mean(loss_1d[:14])
        for i in range(14, len(df_1d)):
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
    
    rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.full_like(avg_gain_1d, np.nan), where=avg_loss_1d!=0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h EMA crossover (9, 21) for entry timing
    close_s = pd.Series(close)
    ema9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(dir_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema9[i]) or
            np.isnan(ema21[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Skip low volume periods
        if volume_ratio < vol_threshold:
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h Supertrend
        trend_up = dir_4h_aligned[i] == 1
        trend_down = dir_4h_aligned[i] == -1
        
        # Determine regime from 1d RSI (avoid extremes)
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 + uptrend + not overbought + volume
            if (ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and 
                trend_up and rsi_not_overbought):
                position = 1
                signals[i] = position_size
            # Short: EMA9 crosses below EMA21 + downtrend + not oversold + volume
            elif (ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and 
                  trend_down and rsi_not_oversold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: EMA9 crosses below EMA21 or trend changes to down
            if (ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]) or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: EMA9 crosses above EMA21 or trend changes to up
            if (ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]) or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Supertrend_RSI_EMA_Volume"
timeframe = "1h"
leverage = 1.0