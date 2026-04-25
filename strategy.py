#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirm
Hypothesis: On 1d timeframe, Donchian(20) breakout combined with 1w EMA34 trend filter and volume confirmation.
Donchian breakouts capture strong momentum moves. 1w EMA34 provides a robust long-term trend filter that avoids whipsaws in bear markets.
Volume confirmation ensures breakouts have institutional participation. Designed for 7-25 trades/year (30-100 over 4 years) to minimize fee drift.
Works in bull markets via breakout continuation and bear markets via filtering out counter-trend breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian(20) channels: upper = 20-period high, lower = 20-period low
    # Using prior 20 periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (completed 20-period lookback)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + 1w EMA (34)
    start_idx = max(20, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 1w EMA34 trend alignment
            long_breakout = curr_high > donchian_upper_aligned[i]
            short_breakout = curr_low < donchian_lower_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA34
            long_trend = curr_close > ema_34_1w_aligned[i]
            short_trend = curr_close < ema_34_1w_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Donchian lower (failed breakout) or trend reverses
            if curr_close < donchian_lower_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Donchian upper (failed breakout) or trend reverses
            if curr_close > donchian_upper_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0