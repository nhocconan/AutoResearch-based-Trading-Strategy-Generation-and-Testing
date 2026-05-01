#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) upper band AND close > 1w EMA50 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below Donchian(20) lower band AND close < 1w EMA50 AND volume > 1.5x 20-period average volume.
# Exit: price crosses Donchian(20) middle band (10-period average of high/low).
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 15-25 trades/year.
# Donchian channels provide clear breakout signals, 1w EMA50 ensures alignment with weekly trend,
# volume confirmation reduces false breakouts. Works in bull (breakouts with trend) and bear (breakdowns with trend).

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
    
    # Calculate Donchian channels (20-period)
    # Upper band: highest high over 20 periods
    # Lower band: lowest low over 20 periods
    # Middle band: average of upper and lower
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper_band[i] = np.max(high[i - lookback + 1:i + 1])
        lower_band[i] = np.min(low[i - lookback + 1:i + 1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)  # warmup for Donchian and 1w EMA50
    
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
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_middle = middle_band[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_vol_threshold = vol_threshold[i]
        
        # Volume confirmation
        volume_confirmed = curr_volume > curr_vol_threshold
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above upper band AND close > 1w EMA50 AND volume confirmed
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1w and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND close < 1w EMA50 AND volume confirmed
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below middle band
            if curr_close < curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above middle band
            if curr_close > curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals