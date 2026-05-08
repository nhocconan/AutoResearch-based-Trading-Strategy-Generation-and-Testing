#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from oversold, price above 1d EMA34, with volume spike.
# Short when Williams %R crosses below -20 from overbought, price below 1d EMA34, with volume spike.
# Exit when Williams %R crosses opposite threshold or trend fails.
# Williams %R (14) identifies overbought/oversold conditions; EMA34 filters trend direction.
# Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year to stay within profitable range.

name = "6h_WilliamsR_Reversal_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R and volume (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h 20-period average volume for volume filter
    vol_ma_20 = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Align all indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 6h bar's volume
            idx_6h = 0
            while idx_6h < len(df_6h) and df_6h.iloc[idx_6h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_6h += 1
            idx_6h -= 1  # last completed 6h bar
            
            if idx_6h >= 0:
                vol_6h_current = df_6h.iloc[idx_6h]['volume']
                vol_filter = vol_6h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R reversal + trend + volume
            # Long when Williams %R crosses above -80 from oversold, price above EMA34, with volume spike
            long_condition = (williams_r_aligned[i] > -80) and (williams_r_aligned[i-1] <= -80) and \
                             ema_34_rising_aligned[i] and (close[i] > ema_34_aligned[i]) and vol_filter
            # Short when Williams %R crosses below -20 from overbought, price below EMA34, with volume spike
            short_condition = (williams_r_aligned[i] < -20) and (williams_r_aligned[i-1] >= -20) and \
                              ema_34_falling_aligned[i] and (close[i] < ema_34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -20 or trend fails
            if (williams_r_aligned[i] < -20) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -80 or trend fails
            if (williams_r_aligned[i] > -80) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals