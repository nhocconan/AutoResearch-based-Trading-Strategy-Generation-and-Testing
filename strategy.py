#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX Trend Filter
# Uses Elder Ray (Bull/Bear Power) for momentum and 12h ADX for trend strength
# Long: Bull Power > 0, Bear Power < 0, 12h ADX > 25 (strong uptrend)
# Short: Bull Power < 0, Bear Power > 0, 12h ADX > 25 (strong downtrend)
# Exit when Elder Ray signals reverse or ADX < 20 (trend weakening)
# Works in bull/bear by only trading strong trends, avoiding whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 12h data for ADX (trend filter)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX calculation (14-period)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA and ADX calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when 12h ADX > 25 (strong trend)
        if adx_12h_aligned[i] < 25:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum)
            if bull_power[i] > 0 and bear_power[i] < 0:
                position = 1
                signals[i] = position_size
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum)
            elif bull_power[i] < 0 and bear_power[i] > 0:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: momentum turns bearish OR trend weakens
            if bull_power[i] < 0 or bear_power[i] > 0 or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: momentum turns bullish OR trend weakens
            if bull_power[i] > 0 or bear_power[i] < 0 or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_12hADX_TrendFilter"
timeframe = "6h"
leverage = 1.0