# 12h_Breakout_BullBear_Regime
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and regime filter (ADX<25 for range, ADX>25 for trend).
# In trending regimes (ADX>25): trade breakouts in direction of trend.
# In ranging regimes (ADX<25): mean-revert at Donchian channels (sell at upper, buy at lower).
# Volume confirmation requires >1.5x 20-bar median volume to avoid false breakouts.
# Designed for low-frequency, high-conviction trades to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=1).max()
        lower = pd.Series(low).rolling(window=window, min_periods=1).min()
        return upper.values, lower.values
    
    upper, lower = donchian_channels(high, low, 20)
    
    # 1-day ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_aligned[i] < 25
        is_trend = adx_aligned[i] >= 25
        
        if is_range:
            # Mean reversion: sell at upper channel, buy at lower channel
            if close[i] >= upper[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25  # short at upper band
            elif close[i] <= lower[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25   # long at lower band
            else:
                signals[i] = signals[i-1] if i > 0 else 0
        else:
            # Trend following: breakout in direction of price
            if close[i] > upper[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25   # long breakout
            elif close[i] < lower[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25  # short breakdown
            else:
                signals[i] = signals[i-1] if i > 0 else 0
    
    return signals

name = "12h_Breakout_BullBear_Regime"
timeframe = "12h"
leverage = 1.0