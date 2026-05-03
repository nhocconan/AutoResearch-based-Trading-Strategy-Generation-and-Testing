#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper channel in bull trend (close > 1w EMA50) with volume > 2x 20-period MA.
# Short when price breaks below Donchian lower channel in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide clear structural breakouts, 1w EMA50 filters for higher timeframe trend alignment,
# and volume confirmation reduces false breakouts. Effective in both bull and bear markets by trading with the 1w trend.

name = "12h_Donchian20_1wEMA50_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need at least 50 for EMA + buffer
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 12h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        dc_upper = donchian_upper[i]
        dc_lower = donchian_lower[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Donchian breakout conditions
        breakout_long = close_val > dc_upper  # Price breaks above upper channel
        breakout_short = close_val < dc_lower  # Price breaks below lower channel
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_long and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_short and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian lower channel OR trend reversal
            if close_val < dc_lower or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian upper channel OR trend reversal
            if close_val > dc_upper or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals