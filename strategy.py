#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter (price > 1w EMA50) and volume confirmation.
# Long when price breaks above 20-period 6h Donchian upper band AND 1w EMA50 trend is up AND volume > 1.5x 20-bar average.
# Short when price breaks below 20-period 6h Donchian lower band AND 1w EMA50 trend is down AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term breakouts.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) via 1w EMA50 trend filter.

name = "6h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_trend = ema_50_1w_aligned[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_vol_ma = vol_ma[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout conditions
        # Long breakout: price closes above upper band
        # Short breakdown: price closes below lower band
        long_breakout = curr_close > curr_upper
        short_breakout = curr_close < curr_lower
        
        # 1w EMA50 trend filter: price above EMA50 = uptrend, below = downtrend
        # We use the 6h close vs 1w EMA50 (aligned) for trend direction
        trend_up = curr_close > curr_ema_trend
        trend_down = curr_close < curr_ema_trend
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: long breakout AND uptrend AND volume confirmation
            if (long_breakout and 
                trend_up and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: short breakdown AND downtrend AND volume confirmation
            elif (short_breakout and 
                  trend_down and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below lower band (breakdown) OR trend turns down
            if (curr_close < curr_lower or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above upper band (breakout) OR trend turns up
            if (curr_close > curr_upper or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals