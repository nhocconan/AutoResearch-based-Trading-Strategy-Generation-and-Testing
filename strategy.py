#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1w trend filter
    # Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA
    # Weekly trend filter ensures we trade with the higher timeframe momentum
    # Volume confirmation filters low-conviction moves
    # Works in bull/bear by adapting to weekly trend while capturing 6h momentum shifts
    # Target: 12-30 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w and 1d data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 13-period EMA on 6h data (Elder Ray base)
    ema_period = 13
    alpha = 2.0 / (ema_period + 1)
    ema_6h = np.full(n, np.nan)
    ema_6h[ema_period-1] = np.mean(close[:ema_period])
    for i in range(ema_period, n):
        ema_6h[i] = alpha * close[i] + (1 - alpha) * ema_6h[i-1]
    
    # Elder Ray components
    bull_power = high - ema_6h  # Buying power
    bear_power = low - ema_6h   # Selling power (negative values indicate selling pressure)
    
    # Smooth Elder Ray with 8-period EMA to reduce noise
    smooth_period = 8
    alpha_smooth = 2.0 / (smooth_period + 1)
    bull_power_smooth = np.full(n, np.nan)
    bear_power_smooth = np.full(n, np.nan)
    
    # Initialize smoothed values
    bull_power_smooth[smooth_period-1] = np.mean(bull_power[:smooth_period])
    bear_power_smooth[smooth_period-1] = np.mean(bear_power[:smooth_period])
    
    for i in range(smooth_period, n):
        bull_power_smooth[i] = alpha_smooth * bull_power[i] + (1 - alpha_smooth) * bull_power_smooth[i-1]
        bear_power_smooth[i] = alpha_smooth * bear_power[i] + (1 - alpha_smooth) * bear_power_smooth[i-1]
    
    # Weekly trend filter: 21-period EMA on weekly close
    ema_21_1w = np.full(len(df_1w), np.nan)
    alpha_1w = 2.0 / (21 + 1)
    ema_21_1w[20] = np.mean(close_1w[:21])
    for i in range(21, len(df_1w)):
        ema_21_1w[i] = alpha_1w * close_1w[i] + (1 - alpha_1w) * ema_21_1w[i-1]
    
    # Weekly trend: 1 if above EMA21, -1 if below
    weekly_trend = np.where(close_1w > ema_21_1w, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # 1d volume filter: volume > 1.2 * 20-day average
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_filter = volume > 1.2 * vol_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Elder Ray divergence with weekly trend filter
        # Long: Bull power rising AND bear power weakening AND weekly uptrend
        # Short: Bear power falling AND bull power weakening AND weekly downtrend
        if i >= 1:
            bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
            bull_falling = bull_power_smooth[i] < bull_power_smooth[i-1]
            bear_rising = bear_power_smooth[i] > bear_power_smooth[i-1]  # Less negative = weakening
            bear_falling = bear_power_smooth[i] < bear_power_smooth[i-1]  # More negative = strengthening
            
            long_entry = bull_rising and (not bear_falling) and weekly_trend_aligned[i] == 1 and volume_filter[i]
            short_entry = bear_falling and (not bull_rising) and weekly_trend_aligned[i] == -1 and volume_filter[i]
        else:
            long_entry = False
            short_entry = False
        
        # Exit conditions: Elder Ray convergence or weekly trend change
        long_exit = (not bull_rising) or bear_falling or weekly_trend_aligned[i] != 1
        short_exit = (not bear_falling) or bull_rising or weekly_trend_aligned[i] != -1
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0