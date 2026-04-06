#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week trend filter and volume confirmation
# Long when price breaks above 20-day high AND 1-week close above 1-week EMA(20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND 1-week close below 1-week EMA(20) AND volume > 1.5x average
# Uses 1-week trend filter to avoid counter-trend trades and volume to confirm breakout strength.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within optimal range.

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Average volume for confirmation (20-period)
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean()
    
    # 1w trend filter: EMA(20) on 1w close
    df_1w = get_htf_data(prices, '1w')
    one_week_close = df_1w['close'].values
    
    # Calculate 20-period EMA on 1w close
    one_week_close_series = pd.Series(one_week_close)
    one_week_ema = one_week_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 1w EMA to 1d timeframe
    one_week_ema_aligned = align_htf_to_ltf(prices, df_1w, one_week_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 1w EMA data not available
        if np.isnan(one_week_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 20-day low OR 1w trend turns bearish
            if (close[i] <= donchian_low[i] or 
                one_week_close[i] < one_week_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-day high OR 1w trend turns bullish
            if (close[i] >= donchian_high[i] or 
                one_week_close[i] > one_week_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above 20-day high AND 1w close above 1w EMA AND volume confirmed
            if (close[i] > donchian_high[i] and 
                one_week_close[i] > one_week_ema_aligned[i] and
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND 1w close below 1w EMA AND volume confirmed
            elif (close[i] < donchian_low[i] and 
                  one_week_close[i] < one_week_ema_aligned[i] and
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
    
    return signals