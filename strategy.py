#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 trending up AND volume > 2x 20-bar avg
# Short when price breaks below Donchian(20) low AND 12h EMA50 trending down AND volume > 2x 20-bar avg
# Exit when price touches Donchian(20) midpoint OR volume drops below average
# Target: 20-50 trades/year via tight entry conditions and trend filter to reduce whipsaw
# Works in bull markets via long breakouts and bear markets via short breakouts
# Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_spike[i]
        ema_trend = ema_50_12h_aligned[i]
        prev_ema_trend = ema_50_12h_aligned[i-1] if i > 0 else ema_trend
        
        # Determine 12h EMA50 trend direction
        trending_up = ema_trend > prev_ema_trend
        trending_down = ema_trend < prev_ema_trend
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND 12h EMA50 trending up AND volume spike
            if close[i] > donchian_high[i] and trending_up and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND 12h EMA50 trending down AND volume spike
            elif close[i] < donchian_low[i] and trending_down and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price touches Donchian mid OR volume drops
            if close[i] <= donchian_mid[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price touches Donchian mid OR volume drops
            if close[i] >= donchian_mid[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals