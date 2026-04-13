#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter + volume confirmation
    # Long: price > upper Donchian(20) AND weekly EMA50 > EMA200 AND volume > 1.5x avg
    # Short: price < lower Donchian(20) AND weekly EMA50 < EMA200 AND volume > 1.5x avg
    # Exit: price crosses middle Donchian (20-period median) OR opposite Donchian touch
    # Using 1d timeframe for optimal trade frequency (target 7-25/year), Donchian for structure,
    # weekly EMA crossover for trend filter, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 and EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMAs to 1d
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 1d Donchian Channels (20-period)
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate 1d volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        bullish_trend = ema50_1w_aligned[i] > ema200_1w_aligned[i]
        bearish_trend = ema50_1w_aligned[i] < ema200_1w_aligned[i]
        
        # Donchian Channel conditions
        donchian_breakout_up = close[i] > upper[i]
        donchian_breakout_down = close[i] < lower[i]
        donchian_middle_cross_up = (close[i] > middle[i]) and (prices['close'].iloc[i-1] <= middle[i-1]) if i > 0 else False
        donchian_middle_cross_down = (close[i] < middle[i]) and (prices['close'].iloc[i-1] >= middle[i-1]) if i > 0 else False
        
        # Entry logic: Donchian breakout + trend alignment + volume confirmation
        long_entry = donchian_breakout_up and bullish_trend and volume_spike[i]
        short_entry = donchian_breakout_down and bearish_trend and volume_spike[i]
        
        # Exit logic: middle Donchian cross or opposite Donchian touch
        long_exit = donchian_middle_cross_down or donchian_breakout_down
        short_exit = donchian_middle_cross_up or donchian_breakout_up
        
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

name = "1d_1w_donchian_breakout_ema_trend_volume_v1"
timeframe = "1d"
leverage = 1.0