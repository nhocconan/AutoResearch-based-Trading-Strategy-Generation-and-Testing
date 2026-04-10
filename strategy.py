#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1w Trend Filter and Volume Spike
# - Primary: 12h timeframe for lower frequency (target: 80-120 trades over 4 years)
# - HTF: 1w for major trend direction (avoid counter-trend trades in bear markets)
# - Long: Price breaks above 20-period Donchian HIGH + 1w close > 1w EMA20 + volume > 2.0x 20-period MA
# - Short: Price breaks below 20-period Donchian LOW + 1w close < 1w EMA20 + volume > 2.0x 20-period MA
# - Exit: Price reverts to 10-period Donchian MIDDLE (mean reversion) or ATR-based stop
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: 1w trend filter prevents shorts in bull markets and longs in bear markets
# - Volume spike ensures breakout legitimacy, reducing false signals

name = "12h_1w_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 12h Donchian Channels (20-period)
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_roll = (high_roll + low_roll) / 2.0  # 10-period middle for exit
    
    # Calculate 1w EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1w trend alignment (close > EMA20 = uptrend, close < EMA20 = downtrend)
    uptrend_1w = close_1w > ema_20_1w
    downtrend_1w = close_1w < ema_20_1w
    
    # Align 1w trend to 12h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Calculate 12h volume moving average (20-period) for volume confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup period (20 for Donchian + 20 for EMA buffer)
        # Skip if any required data is invalid
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(mid_roll[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-period MA
        volume_spike = volume_12h[i] > 2.0 * volume_ma_20_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian HIGH + 1w uptrend + volume spike
            if (close_12h[i] > high_roll[i] and 
                uptrend_1w_aligned[i] > 0.5 and  # 1w uptrend
                volume_spike):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian LOW + 1w downtrend + volume spike
            elif (close_12h[i] < low_roll[i] and 
                  downtrend_1w_aligned[i] > 0.5 and  # 1w downtrend
                  volume_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price reverts to Donchian MIDDLE (mean reversion)
            # 2. Optional: ATR-based stoploss (simplified as opposite Donchian break)
            
            if position == 1:  # Long position
                exit_condition = (
                    close_12h[i] < mid_roll[i]  # Reverted to middle (mean reversion)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    close_12h[i] > mid_roll[i]  # Reverted to middle (mean reversion)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals