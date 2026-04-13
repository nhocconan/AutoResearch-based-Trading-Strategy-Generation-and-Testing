#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with daily volume confirmation and ADX trend filter.
# Uses Williams %R(14) on 4h for overbought/oversold signals, confirmed by daily volume spike
# and filtered by daily ADX to avoid ranging markets. Designed to work in both bull and bear
# markets by fading extremes in the direction of the higher timeframe trend.
# Target: 80-160 total trades over 4 years (20-40/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate 4-hour Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4-hour timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.8x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.8)
        
        # ADX condition: trending market (ADX > 25)
        adx_condition = adx_aligned[i] > 25
        
        # Williams %R conditions: extreme levels for mean reversion
        williams_oversold = williams_r_aligned[i] < -80  # Oversold
        williams_overbought = williams_r_aligned[i] > -20  # Overbought
        
        # Entry conditions: Williams %R extreme with volume and trend filter
        # Long when Williams %R < -80 (oversold) with volume and uptrend
        # Short when Williams %R > -20 (overbought) with volume and downtrend
        if position == 0:
            if williams_oversold and volume_condition and adx_condition:
                # Additional check: only go long if DI+ > DI- (uptrend)
                if 100 * dm_plus_14[i] / tr_14[i] > 100 * dm_minus_14[i] / tr_14[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif williams_overbought and volume_condition and adx_condition:
                # Additional check: only go short if DI- > DI+ (downtrend)
                if 100 * dm_minus_14[i] / tr_14[i] > 100 * dm_plus_14[i] / tr_14[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when Williams %R returns to neutral or volume drops
            if williams_r_aligned[i] > -50 or volume[i] <= volume_4h_approx_ma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when Williams %R returns to neutral or volume drops
            if williams_r_aligned[i] < -50 or volume[i] <= volume_4h_approx_ma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_MeanReversion_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0