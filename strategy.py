#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Long when price breaks above 1w Donchian upper + volume spike + price > 1w EMA(34)
# Short when price breaks below 1w Donchian lower + volume spike + price < 1w EMA(34)
# Uses weekly Donchian channels from prior week to avoid look-ahead
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Designed for low trade frequency (7-25/year on 1d) to minimize fee drag
# Works in bull (breakouts with trend) and bear (breakdowns with trend) markets

name = "1d_Donchian20_Volume_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels and EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from prior 1w session (20-period high/low)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 34, 20)  # 1w EMA(34), Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume spike + price > 1w EMA(34)
            if (close[i] > donchian_upper_aligned[i] and volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + volume spike + price < 1w EMA(34)
            elif (close[i] < donchian_lower_aligned[i] and volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price below Donchian lower or price below 1w EMA(34)
            if (close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price above Donchian upper or price above 1w EMA(34)
            if (close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals