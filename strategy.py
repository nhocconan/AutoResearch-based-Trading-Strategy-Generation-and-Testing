#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA50 for primary trend direction (bull/bear) to capture major regimes.
# Long when: price breaks above Donchian(20) high AND 1w EMA50 trending up AND volume > 1.5x 20-period average.
# Short when: price breaks below Donchian(20) low AND 1w EMA50 trending down AND volume > 1.5x 20-period average.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 15-25 trades/year.
# Donchian channels provide clear breakout levels; 1w EMA50 filters counter-trend trades in strong markets.
# Volume confirmation ensures breakouts have conviction. Works in bull (trend following) and bear (avoiding fakeouts).

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(lookback-1, n):
        vol_avg[i] = np.mean(volume[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback-1  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_vol_avg = vol_avg[i]
        
        # Trend conditions
        ema_trend_up = curr_ema_50_1w > np.roll(ema_50_1w_aligned, 1)[i] if i > 0 else False
        ema_trend_down = curr_ema_50_1w < np.roll(ema_50_1w_aligned, 1)[i] if i > 0 else False
        
        # Volume confirmation
        volume_confirm = curr_volume > 1.5 * curr_vol_avg
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high + uptrend + volume confirmation
            if (curr_close > curr_highest_high and 
                ema_trend_up and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + downtrend + volume confirmation
            elif (curr_close < curr_lowest_low and 
                  ema_trend_down and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if (curr_close < curr_lowest_low or not ema_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if (curr_close > curr_highest_high or not ema_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals