#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation.
# In low volatility periods, price breaks out of Bollinger Bands with continuation.
# Uses 1-week EMA200 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 12-37 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(200) on weekly close
    ema_200_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (200 + 1)
    for i in range(len(close_1w)):
        if i < 199:
            ema_200_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_200_1w[i-1]):
                ema_200_1w[i] = np.mean(close_1w[i-199:i+1])
            else:
                ema_200_1w[i] = close_1w[i] * alpha + ema_200_1w[i-1] * (1 - alpha)
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Bollinger Bands (20, 2) on 12h data
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        rolling_mean[i] = np.mean(close[i-bb_period+1:i+1])
        rolling_std[i] = np.std(close[i-bb_period+1:i+1])
    
    upper_band = rolling_mean + (bb_std * rolling_std)
    lower_band = rolling_mean - (bb_std * rolling_std)
    
    # Bollinger Band width (normalized)
    bb_width = np.zeros(n)
    for i in range(n):
        if not np.isnan(rolling_mean[i]) and rolling_mean[i] != 0:
            bb_width[i] = (upper_band[i] - lower_band[i]) / rolling_mean[i]
        else:
            bb_width[i] = np.nan
    
    # Bollinger Band width percentile (50-period)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback - 1, n):
        if not np.isnan(bb_width[i-lookback+1:i+1]).any():
            bb_width_percentile[i] = np.percentile(bb_width[i-lookback+1:i+1], 20)  # 20th percentile = squeeze
    
    # Volume spike: current volume > 1.8 * 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(bb_period, lookback, 30, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA200
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_200_1w_aligned[i-1]):
            trend_up = ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1]
            trend_down = ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        # Squeeze condition: BB width below 20th percentile of last 50 periods
        is_squeeze = bb_width_percentile[i] > bb_width[i] if not np.isnan(bb_width_percentile[i]) else False
        
        if position == 0:
            # Long entry: Bollinger Band breakout up + uptrend + volume spike + squeeze
            if (close[i] > upper_band[i] and 
                trend_up and 
                volume_spike[i] and
                is_squeeze):
                signals[i] = 0.25
                position = 1
            # Short entry: Bollinger Band breakout down + downtrend + volume spike + squeeze
            elif (close[i] < lower_band[i] and 
                  trend_down and 
                  volume_spike[i] and
                  is_squeeze):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle band or trend turns down
            if (close[i] < rolling_mean[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band or trend turns up
            if (close[i] > rolling_mean[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_Breakout_1wEMA200_Volume_v1"
timeframe = "12h"
leverage = 1.0