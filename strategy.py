#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses Donchian channel breakouts for trend capture, 1w EMA50 for higher timeframe trend direction,
# volume confirmation (>1.5x 20-bar avg) to reduce false breakouts.
# Discrete position sizing at ±0.30 to manage fee drag.
# Target: 50-100 total trades over 4 years (12-25/year) to avoid excessive fees on 1d timeframe.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_vals = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20) on 1d
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above upper channel, close > 1w EMA50, volume spike
            if (curr_high > upper_channel[i] and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: Donchian breakout below lower channel, close < 1w EMA50, volume spike
            elif (curr_low < lower_channel[i] and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle of channel (mean reversion) or trend reversal
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if curr_close < middle_channel:  # Price back below channel middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle of channel
            middle_channel = (upper_channel[i] + lower_channel[i]) / 2
            if curr_close > middle_channel:  # Price back above channel middle
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals