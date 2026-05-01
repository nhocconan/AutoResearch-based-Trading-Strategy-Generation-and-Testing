#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1w EMA50 for primary trend direction to avoid counter-trend trades.
# Long when: price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-period average volume.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 15-25 trades/year.
# Donchian channels provide structural breakout signals; 1w EMA50 ensures alignment with higher timeframe trend.
# Volume confirmation filters false breakouts. Works in bull (trend following) and bear (avoiding false signals in ranging markets).

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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(19, n):
        avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 49, 19)  # warmup for Donchian, EMA, and volume
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_50 = ema_50_1w_aligned[i]
        curr_avg_volume = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = curr_volume > (1.5 * curr_avg_volume)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price > 1w EMA50 AND volume confirmed
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_50 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1w EMA50 AND volume confirmed
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_50 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low (stoploss) OR price < 1w EMA50 (trend change)
            if (curr_close < curr_donchian_low or 
                curr_close < curr_ema_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (stoploss) OR price > 1w EMA50 (trend change)
            if (curr_close > curr_donchian_high or 
                curr_close > curr_ema_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals