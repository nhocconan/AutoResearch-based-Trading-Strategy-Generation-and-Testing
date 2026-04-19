#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 1d ADX regime filter
# - Elder Ray Bull Power = High - EMA(13), Bear Power = EMA(13) - Low (on 6h)
# - 1d ADX > 25 indicates trending market, < 20 indicates ranging
# - In trending (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging (ADX<20): fade extreme Elder Ray values (long when Bear Power < -std, short when Bull Power < -std)
# - Volume confirmation: 6h volume > 1.5x 20-period average
# - Position size: 0.25 to manage drawdown
# - Designed to work in both bull and bear markets by adapting to regime

name = "6h_ElderRay_1dADXRegime_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX calculation
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align ADX to 6h
    adx_6h_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Elder Ray on 6h: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Regime thresholds
    adx_trending = 25
    adx_ranging = 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_6h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_6h[i]) or
            np.isnan(ema_13[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Determine regime
            if adx_6h_aligned[i] > adx_trending:  # Trending market
                # Long: Bull Power positive and rising
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power positive and rising
                elif bear_power[i] > 0 and bear_power[i] > bear_power[i-1] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            elif adx_6h_aligned[i] < adx_ranging:  # Ranging market
                # Calculate standard deviation of Elder Ray for extreme detection
                # Use rolling std of Bear Power for long signals, Bull Power for short signals
                bear_std = pd.Series(bear_power[max(0, i-20):i+1]).std()
                bull_std = pd.Series(bull_power[max(0, i-20):i+1]).std()
                
                # Long: Bear Power extremely negative (strong selling exhaustion)
                if bear_std > 0 and bear_power[i] < -1.5 * bear_std and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power extremely negative (strong buying exhaustion)
                elif bull_std > 0 and bull_power[i] < -1.5 * bull_std and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit conditions
            if adx_6h_aligned[i] > adx_trending:
                # In trending: exit when Bull Power turns negative
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging: exit when Bear Power recovers from extreme
                bear_std = pd.Series(bear_power[max(0, i-20):i+1]).std()
                if bear_std > 0 and bear_power[i] > -0.5 * bear_std:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short exit conditions
            if adx_6h_aligned[i] > adx_trending:
                # In trending: exit when Bear Power turns negative
                if bear_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging: exit when Bull Power recovers from extreme
                bull_std = pd.Series(bull_power[max(0, i-20):i+1]).std()
                if bull_std > 0 and bull_power[i] > -0.5 * bull_std:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals