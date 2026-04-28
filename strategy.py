#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike filter
# Long when price breaks above 20-period 12h high AND close > 1d EMA50 AND volume > 2.5x 20-bar avg
# Short when price breaks below 20-period 12h low AND close < 1d EMA50 AND volume > 2.5x 20-bar avg
# Exit when price retraces to 10-period 12h opposite Donchian level (mean reversion within channel)
# Uses discrete sizing (0.25) to minimize fee churn. Target: 15-30 trades/year on 12h.
# Works in bull markets via trend-following breakouts, works in bear via volume spike exhaustion signals
# that often precede reversals after panic moves. EMA50 filter ensures we trade with higher timeframe trend.

name = "12h_Donchian20_EMA50_VolumeSpike_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values  # for exits
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values   # for exits
    
    # Volume confirmation: >2.5x 20-bar average volume (strict filter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        ema_trend = ema_50_1d_aligned[i]
        dh20 = donchian_high_20[i]
        dl20 = donchian_low_20[i]
        dh10 = donchian_high_10[i]
        dl10 = donchian_low_10[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above 20-period high AND close > 1d EMA50 AND volume spike
            if curr_close > dh20 and curr_close > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period low AND close < 1d EMA50 AND volume spike
            elif curr_close < dl20 and curr_close < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retraces to 10-period low (mean reversion)
            if curr_close <= dl10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retraces to 10-period high (mean reversion)
            if curr_close >= dh10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals