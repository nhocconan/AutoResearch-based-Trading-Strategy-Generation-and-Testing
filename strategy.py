#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power Regime with 1d ADX filter and 1w trend alignment
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power rising (improving) AND 1d ADX > 25 (trending) AND 1w close > 1w open
# - Short when Bear Power > 0 AND Bull Power falling (worsening) AND 1d ADX > 25 AND 1w close < 1w open
# - Exit when power signal reverses or ADX < 20 (range regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures bull/bear strength relative to EMA; ADX filters for trending markets only
# - Weekly trend filter ensures alignment with higher timeframe momentum
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_elder_ray_power_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute EMA13 for Elder Ray (13-period EMA of close)
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr == 0, 1, atr)
    di_minus = 100 * dm_minus_smooth / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Power signals: rising/falling power (compare to previous bar)
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_rising = i > 0 and bear_power[i] > bear_power[i-1]
        bull_power_falling = i > 0 and bull_power[i] < bull_power[i-1]
        bear_power_falling = i > 0 and bear_power[i] < bear_power[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND rising AND ADX > 25 AND weekly bullish
            if (bull_power[i] > 0 and bull_power_rising and 
                adx_aligned[i] > 25 and weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 AND rising AND ADX > 25 AND weekly bearish
            elif (bear_power[i] > 0 and bear_power_rising and 
                  adx_aligned[i] > 25 and weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit conditions:
            # 1. Power signal reverses (Bull Power <= 0 for long, Bear Power <= 0 for short)
            # 2. Power starts falling (weakening momentum)
            # 3. ADX < 20 (market going into range)
            exit_signal = False
            
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or not bull_power_rising or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            elif position == -1:  # Short position
                if (bear_power[i] <= 0 or not bear_power_rising or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals