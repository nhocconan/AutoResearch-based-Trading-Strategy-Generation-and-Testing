#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper channel AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower channel AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses Donchian midline (20-period average of high/low).
# Uses discrete position sizing (0.30) to limit drawdown and fee churn.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via 12h EMA50 trend filter.

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20-period)
    # Upper channel: highest high over 20 periods
    # Lower channel: lowest low over 20 periods
    # Middle channel: average of upper and lower
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above upper channel, uptrend (price > 12h EMA50), volume confirmation
            if (curr_high > highest_high[i] and 
                curr_close > ema_50_12h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: break below lower channel, downtrend (price < 12h EMA50), volume confirmation
            elif (curr_low < lowest_low[i] and 
                  curr_close < ema_50_12h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian middle
            if curr_close < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian middle
            if curr_close > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals