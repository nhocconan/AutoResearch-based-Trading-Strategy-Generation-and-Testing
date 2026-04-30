#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume spike + session
# Williams %R(14) identifies overbought/oversold conditions. In ranging markets,
# fade extreme readings (> -20 for short, < -80 for long). In trending markets
# (ADX > 25), breakout continuation when price breaks Donchian(20) with volume.
# Uses 1d EMA34 for trend direction and 6h ADX(14) for regime detection.
# Discrete sizing ±0.25 to limit drawdown. Target: 60-120 trades over 4 years (15-30/year).

name = "6h_WilliamsR_1dEMA34_ADXRegime_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h
    williams_period = 14
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # ADX(14) for regime detection on 6h
    adx_period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    atr_smooth = pd.Series(atr).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    # Avoid division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=adx_period, adjust=False, min_periods=adx_period).mean().values
    
    # Donchian(20) for breakout signals (use shift(1) to avoid look-ahead)
    donchian_period = 20
    highest_20 = pd.Series(high).shift(1).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_20 = pd.Series(low).shift(1).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, williams_period, adx_period, donchian_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(adx[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_adx = adx[i]
        curr_upper = highest_20[i]
        curr_lower = lowest_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
            if curr_adx > 25:  # Trending market
                # Breakout continuation: price breaks Donchian with volume and trend alignment
                if (curr_close > curr_upper and 
                    curr_close > curr_ema_34_1d and 
                    curr_volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (curr_close < curr_lower and 
                      curr_close < curr_ema_34_1d and 
                      curr_volume_confirm):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Mean reversion: fade Williams %R extremes
                if (curr_williams < -80 and  # Oversold
                    curr_close > curr_ema_34_1d and  # Above trend filter
                    curr_volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (curr_williams > -20 and  # Overbought
                      curr_close < curr_ema_34_1d and  # Below trend filter
                      curr_volume_confirm):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: reverse signal or stoploss
            # Stoploss: 2.5 * ATR below entry (simplified as Donchian lower break)
            if curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: reverse signal or stoploss
            # Stoploss: 2.5 * ATR above entry (simplified as Donchian upper break)
            if curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals