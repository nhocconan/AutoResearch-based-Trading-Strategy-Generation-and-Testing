#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian channel breakout with daily trend filter + volume spike
    # Long: Price breaks above Donchian(20) high AND daily close > daily EMA50 AND volume > 2.0x 20-period avg
    # Short: Price breaks below Donchian(20) low AND daily close < daily EMA50 AND volume > 2.0x 20-period avg
    # Exit: Price crosses Donchian middle (mean of 20-period high/low) OR opposite Donchian touch
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), Donchian for structure,
    # daily EMA50 for trend filter, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        donchian_high[i] = np.max(high[i-donchian_period:i])
        donchian_low[i] = np.min(low[i-donchian_period:i])
    
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Get 4h volume for confirmation (>2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_middle[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        bullish_trend = close_1d[i // 16] > ema50_1d[i // 16] if i // 16 < len(close_1d) else False
        bearish_trend = close_1d[i // 16] < ema50_1d[i // 16] if i // 16 < len(close_1d) else False
        
        # Donchian Channel conditions
        donchian_breakout_up = close[i] > donchian_high[i]
        donchian_breakout_down = close[i] < donchian_low[i]
        donchian_middle_cross_up = (close[i] > donchian_middle[i]) and (prices['close'].iloc[i-1] <= donchian_middle[i-1]) if i > 0 else False
        donchian_middle_cross_down = (close[i] < donchian_middle[i]) and (prices['close'].iloc[i-1] >= donchian_middle[i-1]) if i > 0 else False
        
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

name = "4h_1d_donchian_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0