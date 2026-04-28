#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX25 regime filter and volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising (2-bar momentum) with price > 1d EMA34 (uptrend) and volume spike.
# Short when Bear Power < 0 and falling (2-bar momentum) with price < 1d EMA34 (downtrend) and volume spike.
# Uses 1d ADX > 25 to filter for trending regimes only, avoiding whipsaws in ranging markets.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 1d EMA34 trend filter and ADX regime filter.

name = "6h_ElderRay_1dADX25_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ADX for regime filter (trending when ADX > 25)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx[np.isnan(dx)] = 0  # Handle division by zero
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trending_regime = adx_aligned > 25
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Elder Ray momentum (2-bar change)
    bull_power_momentum = bull_power - np.roll(bull_power, 2)
    bear_power_momentum = bear_power - np.roll(bear_power, 2)
    bull_power_momentum[:2] = 0
    bear_power_momentum[:2] = 0
    
    # Volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(bull_power_momentum[i]) or 
            np.isnan(bear_power_momentum[i]) or 
            np.isnan(volume_ma_20[i]) or 
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray conditions with momentum
        long_signal = (bull_power[i] > 0 and 
                      bull_power_momentum[i] > 0 and 
                      price_above_ema and 
                      volume_spike[i] and 
                      trending_regime[i])
        short_signal = (bear_power[i] < 0 and 
                       bear_power_momentum[i] < 0 and 
                       price_below_ema and 
                       volume_spike[i] and 
                       trending_regime[i])
        
        # Exit conditions: opposite Elder Ray signal or trend reversal
        long_exit = (bull_power[i] < 0) or (close[i] < ema_34_1d_aligned[i])
        short_exit = (bear_power[i] > 0) or (close[i] > ema_34_1d_aligned[i])
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals