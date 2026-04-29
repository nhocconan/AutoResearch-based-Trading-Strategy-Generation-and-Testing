#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Uses 4h EMA for trend direction, 1h Donchian channels for breakout timing,
# volume spike for confirmation, and session filter (08-20 UTC) to reduce noise.
# Target: 15-35 trades/year (60-140 total over 4 years) by requiring confluence
# of trend, breakout, volume, and session - avoids overtrading while capturing
# strong momentum moves in both bull and bear markets.

name = "1h_DonchianBreakout_4hEMATrend_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 4h EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_4h = ema_4h_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper + 4h uptrend + volume
            if (curr_high > curr_highest_high and 
                curr_close > curr_ema_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower + 4h downtrend + volume
            elif (curr_low < curr_lowest_low and 
                  curr_close < curr_ema_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below Donchian lower OR 4h trend turns down
            if (curr_low < curr_lowest_low or 
                curr_close < curr_ema_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above Donchian upper OR 4h trend turns up
            if (curr_high > curr_highest_high or 
                curr_close > curr_ema_4h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals