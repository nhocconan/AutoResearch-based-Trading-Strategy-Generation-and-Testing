#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when: price breaks above Donchian upper (20-period high) AND 1w close > 1w EMA50 AND 1d volume > 1.5x 20-period average
# Short when: price breaks below Donchian lower (20-period low) AND 1w close < 1w EMA50 AND 1d volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 15-25 trades/year on 1d.
# Donchian provides clear breakout structure, 1w EMA50 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (catching breakouts with trend) and bear (catching breakdowns with trend) by trading with the aligned trend.

name = "1d_Donchian20_1wTrend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    donch_hi = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume average (20-period) for volume spike confirmation
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (1d timeframe, full day active)
        hour = hours[i]
        in_session = True  # 1d uses full day
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or np.isnan(vol_ma_1d[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_1d[i]
        curr_donch_hi = donch_hi[i]
        curr_donch_lo = donch_lo[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_close > curr_donch_hi
        breakout_down = curr_close < curr_donch_lo
        
        # 1w trend filter: price above/below EMA50
        uptrend_1w = curr_close > curr_ema_50_1w
        downtrend_1w = curr_close < curr_ema_50_1w
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND 1w uptrend AND volume spike
            if (breakout_up and 
                uptrend_1w and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND 1w downtrend AND volume spike
            elif (breakout_down and 
                  downtrend_1w and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian lower (stoploss/reversal)
            if curr_close < curr_donch_lo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper (stoploss/reversal)
            if curr_close > curr_donch_hi:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals