#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) + volume spike + price > 1w EMA(34)
# Short when price breaks below Donchian lower (20) + volume spike + price < 1w EMA(34)
# Uses Donchian levels from previous 1d bar to avoid look-ahead
# 1w EMA(34) filter captures long-term trend and reduces whipsaw in bear markets
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for low trade frequency (7-25/year on 1d) to minimize fee drag
# Works in both bull (breakouts) and bear (mean reversion at extremes) markets

name = "1d_Donchian20_Volume_1wEMA34_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20) from previous 1d bar
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's high, low to calculate today's Donchian levels
        high_prev = high[i-1]
        low_prev = low[i-1]
        
        donchian_upper[i] = high_prev
        donchian_lower[i] = low_prev
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(1 for Donchian, 20 for volume MA, 34 for 1w EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume spike + price > 1w EMA(34)
            if (close[i] > donchian_upper[i] and volume_spike[i] and close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + volume spike + price < 1w EMA(34)
            elif (close[i] < donchian_lower[i] and volume_spike[i] and close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR price below 1w EMA(34)
            if (close[i] < donchian_lower[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR price above 1w EMA(34)
            if (close[i] > donchian_upper[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals