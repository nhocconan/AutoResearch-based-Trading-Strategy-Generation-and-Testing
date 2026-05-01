#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d ADX trend filter and volume spike confirmation.
# Long when: Williams %R(14) < -80 (oversold) AND price closes above prior bar's high AND 1d ADX > 25 AND volume > 1.5x 20-bar average
# Short when: Williams %R(14) > -20 (overbought) AND price closes below prior bar's low AND 1d ADX > 25 AND volume > 1.5x 20-bar average
# Exit when: Williams %R crosses above -50 (for long) or below -50 (for short) OR ADX < 20 (trend weakening)
# Uses Williams %R for mean reversion extremes, 1d ADX for trend strength, volume spike for conviction.
# Target: 20-40 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing reversals in trending markets.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) by trading with aligned 1d trend.

name = "4h_WilliamsR_1dADX_VolumeReversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h Williams %R (14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX (14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Calculate Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smooth TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero (when DI+ and DI- are both zero)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h primary timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Williams %R (14) + ADX (14+14)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_williams_r = williams_r_aligned[i]
        curr_adx = adx_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        # Calculate 4h volume MA on the fly using aligned 4h data
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Prior bar high/low for breakout confirmation
        prior_high = high[i-1] if i > 0 else high[i]
        prior_low = low[i-1] if i > 0 else low[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) + close above prior high + ADX > 25 + volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > prior_high and 
                curr_adx > 25 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + close below prior low + ADX > 25 + volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < prior_low and 
                  curr_adx > 25 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion) OR ADX < 20 (trend weakening)
            if (curr_williams_r > -50) or \
               (curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion) OR ADX < 20 (trend weakening)
            if (curr_williams_r < -50) or \
               (curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals