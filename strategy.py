#!/usr/bin/env python3
# 1h_4h1d_donchian_vol_regime_v1
# Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter (price > SMA50) and volatility regime filter.
# Enters long when price breaks above 4h Donchian(20) upper band, price > 1d SMA50, and ATR(14) < ATR(50) (low volatility regime).
# Enters short when price breaks below 4h Donchian(20) lower band, price < 1d SMA50, and ATR(14) < ATR(50).
# Uses discrete position sizing (±0.20) to minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year). Works in bull/bear via Donchian structure and trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data ONCE before loop for Donchian
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels for 4h (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = rolling max of high, lower = rolling min of low
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (completed 4h candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Get 1d HTF data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align SMA50 to 1h timeframe (completed daily candle only)
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # ATR regime filter: ATR(14) < ATR(50) indicates low volatility (good for breakouts)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr2[0] = tr1[0]  # first bar
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr14 < atr50
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(sma50_1d_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 4h Donchian lower band OR trend breaks
            if (close[i] < donchian_lower_aligned[i]) or (close[i] < sma50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above 4h Donchian upper band OR trend breaks
            if (close[i] > donchian_upper_aligned[i]) or (close[i] > sma50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above 4h Donchian upper, above 1d SMA50, low vol regime
            if (close[i] > donchian_upper_aligned[i]) and (close[i] > sma50_1d_aligned[i]) and low_vol_regime[i]:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below 4h Donchian lower, below 1d SMA50, low vol regime
            elif (close[i] < donchian_lower_aligned[i]) and (close[i] < sma50_1d_aligned[i]) and low_vol_regime[i]:
                position = -1
                signals[i] = -0.20
    
    return signals