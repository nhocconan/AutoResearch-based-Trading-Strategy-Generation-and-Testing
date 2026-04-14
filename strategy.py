#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Weekly Williams %R Reversal with Volume Filter
# Uses ADX(14) to identify trending conditions and Williams %R(14) on weekly chart for reversals
# In trending markets (ADX > 25), extreme weekly Williams %R (< -80 or > -20) signals reversal
# Volume confirmation ensures institutional participation
# Works in both bull/bear markets by trading reversals within established trends
# Target: 20-35 trades/year (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Williams %R
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Williams %R (14-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_weekly).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_weekly).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_weekly) / (highest_high - lowest_low)) * -100,
        -50  # neutral when no range
    )
    williams_r_aligned = align_htf_to_ltf(prices, df_weekly, williams_r)
    
    # Load daily data for ADX calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period) on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily),
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    dm_minus = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)),
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume moving average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for Williams %R and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        strong_trend = adx_aligned[i] > 25
        volume_confirm = volume[i] > vol_ma[i]  # above average volume
        
        if position == 0:
            # Long: weekly Williams %R oversold (< -80) in uptrend with volume
            if williams_r_aligned[i] < -80 and strong_trend and volume_confirm:
                # Additional check: price above weekly EMA for trend alignment
                weekly_ema = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
                weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
                if not np.isnan(weekly_ema_aligned[i]) and close[i] > weekly_ema_aligned[i]:
                    position = 1
                    signals[i] = position_size
            # Short: weekly Williams %R overbought (> -20) in downtrend with volume
            elif williams_r_aligned[i] > -20 and strong_trend and volume_confirm:
                # Additional check: price below weekly EMA for trend alignment
                weekly_ema = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
                weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
                if not np.isnan(weekly_ema_aligned[i]) and close[i] < weekly_ema_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral area or trend weakens
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral area or trend weakens
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX_WeeklyWilliamsR_Volume"
timeframe = "6h"
leverage = 1.0