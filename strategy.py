#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike
Hypothesis: On 1h timeframe, price breaking Camarilla R1/S1 levels from the prior 4h bar, in the direction of the 4h EMA50 trend with 1d volume confirmation (>2.0x 20-period MA), captures high-probability intraday trend continuation. The 4h EMA50 filters for intermediate-term trend, while 1d volume spike confirms broad market participation. Designed for 15-37 trades/year on BTC/ETH with discrete sizing (±0.20) and no trailing stop (reliance on strict entry conditions) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla calculation and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous 4h bar's range
    daily_range = high_4h - low_4h
    camarilla_r1 = close_4h + daily_range * 1.1 / 12
    camarilla_s1 = close_4h - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    close_series = pd.Series(close_4h)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA(20) for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA (50), volume MA (20) + time for 4h and 1d alignment
    start_idx = max(50, 20) + 4  # +4 to ensure 4h bar completion (1h -> 4h: 4 bars)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            # Hold current position or stay flat
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
            
        close_val = close[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_val) or 
            np.isnan(volume_spike)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike
        short_entry = trend_bearish and short_breakout and vol_spike
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < s1_val or not trend_bullish):
            # Exit long: price breaks S1 or trend turns bearish
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > r1_val or not trend_bearish):
            # Exit short: price breaks R1 or trend turns bullish
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0