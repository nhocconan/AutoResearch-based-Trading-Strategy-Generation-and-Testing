#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-day Donchian high with 1w uptrend (close > 1w EMA50) and volume > 1.8x 20-bar avg.
# Short when price breaks below 20-day Donchian low with 1w downtrend (close < 1w EMA50) and volume > 1.8x 20-bar avg.
# Exit on opposite Donchian level touch (mean reversion within the channel).
# Uses proven Donchian breakout structure with strict volume confirmation and 1w EMA50 trend filter to limit trades.
# 1w EMA50 provides longer-term trend filter, reducing false signals in choppy markets and bear rallies.
# Timeframe: 1d, HTF: 1w as per experiment guidelines.

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels - 20-period high/low
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average (to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high_max = high_rolling_max[i]
        curr_low_min = low_rolling_min[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-day Donchian high, uptrend (close > 1w EMA50), volume spike
            if (curr_close > curr_high_max and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day Donchian low, downtrend (close < 1w EMA50), volume spike
            elif (curr_close < curr_low_min and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches 20-day Donchian low (mean reversion)
            if curr_close <= curr_low_min:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches 20-day Donchian high (mean reversion)
            if curr_close >= curr_high_max:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals