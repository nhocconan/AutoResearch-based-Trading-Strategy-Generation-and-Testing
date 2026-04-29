#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Long when price breaks above upper Donchian band, 1w EMA50 up-trend, volume > 2.0x average
# Short when price breaks below lower Donchian band, 1w EMA50 down-trend, volume > 2.0x average
# Exit when price crosses the opposite Donchian band (full reversal)
# Uses discrete position sizing (0.30) and strong volume filter to target ~10-20 trades/year.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Daily timeframe minimizes fee drag while capturing major moves.

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Donchian bands (based on previous period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d Donchian bands using previous 20 periods
    high_prev = df_1d['high'].shift(1).rolling(window=20, min_periods=20).max().values
    low_prev = df_1d['low'].shift(1).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 1d timeframe (no additional delay needed for Donchian)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_prev)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_prev)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume and 1w EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_upper = upper_band_aligned[i]
        curr_lower = lower_band_aligned[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below lower Donchian band (trend reversal)
            if curr_low < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price above upper Donchian band (trend reversal)
            if curr_high > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strong filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above upper band, 1w EMA50 up-trend, volume confirmed
            if curr_high > curr_upper and curr_close > curr_ema50_1w and vol_confirmed:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below lower band, 1w EMA50 down-trend, volume confirmed
            elif curr_low < curr_lower and curr_close < curr_ema50_1w and vol_confirmed:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals