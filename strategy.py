#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with 1w volume filter and 1d ATR volatility filter
# - Uses 1d Donchian(20) channels for breakout signals (structure)
# - Uses 1w average volume > 1.5x 20-period average for volume confirmation (institutional interest)
# - Uses 1d ATR(14) < 30-day ATR mean for low volatility regime (avoid choppy markets)
# - Enters long on breakout above 1d upper band with volume confirmation and low vol
# - Enters short on breakout below 1d lower band with volume confirmation and low vol
# - Exits when price returns to 1d middle (mean reversion within the channel)
# - Designed to capture institutional breakouts after low volatility periods with volume confirmation
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_1dDonchian_1wVolume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d ATR(14) for volatility filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 30-day ATR mean for volatility regime filter
    atr_ma_30 = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    low_volatility = atr < atr_ma_30  # Low vol when current ATR < 30-day average
    
    # Calculate 1w average volume for volume filter
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_1w > (1.5 * vol_ma_20_1w)  # Volume > 1.5x 20-period average
    
    # Align 1d indicators to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_4h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_4h = align_htf_to_ltf(prices, df_1d, donchian_middle)
    low_volatility_4h = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    # Align 1w volume filter to 4h timeframe
    volume_filter_4h = align_htf_to_ltf(prices, df_1w, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_upper_4h[i]) or np.isnan(donchian_lower_4h[i]) or 
            np.isnan(donchian_middle_4h[i]) or np.isnan(low_volatility_4h[i]) or 
            np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for low volatility regime and volume confirmation
            if low_volatility_4h[i] and volume_filter_4h[i]:
                # Long: breakout above upper band
                if close[i] > donchian_upper_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: breakdown below lower band
                elif close[i] < donchian_lower_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: return to middle band (mean reversion)
            if close[i] > donchian_middle_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle band (mean reversion)
            if close[i] < donchian_middle_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals