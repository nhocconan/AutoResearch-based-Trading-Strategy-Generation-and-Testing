#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# Williams %R (14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# In trending markets (price > 1d EMA50): fade extremes (sell at > -20, buy at < -80)
# In ranging markets (price near 1d EMA50): mean revert at Bollinger Bands (20, 2)
# Volume confirmation: require volume > 1.5x 20-period EMA to avoid false signals
# Designed for 12-25 trades/year with proper risk control in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Highest high and lowest low over 14 periods
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    
    # Williams %R: -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    willr = willr.replace([np.inf, -np.inf], np.nan).fillna(-50).values  # Handle division by zero
    
    # Calculate 1d EMA (50-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bollinger Bands (20, 2) on 12h for ranging market entries
    sma_20 = close_series.rolling(window=20, min_periods=20).mean()
    std_20 = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d EMA
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)[i]
        
        if np.isnan(willr[i]) or np.isnan(ema_1d_aligned) or np.isnan(vol_ma[i]) or \
           np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Market regime: trending if price > 1d EMA50, ranging if near EMA50
        price_vs_ema = (close[i] - ema_1d_aligned) / ema_1d_aligned
        is_trending = abs(price_vs_ema) > 0.02  # >2% deviation from EMA = trending
        is_ranging = abs(price_vs_ema) <= 0.02   # Within 2% of EMA = ranging
        
        if is_trending:  # Trending market - fade Williams %R extremes
            if position == 0 and willr[i] < -80 and volume_confirm:  # Oversold -> buy
                position = 1
                signals[i] = position_size
            elif position == 0 and willr[i] > -20 and volume_confirm:  # Overbought -> sell
                position = -1
                signals[i] = -position_size
            elif position == 1 and willr[i] > -50:  # Exit on mean reversion
                position = 0
                signals[i] = 0.0
            elif position == -1 and willr[i] < -50:  # Exit on mean reversion
                position = 0
                signals[i] = 0.0
        elif is_ranging:  # Ranging market - mean revert at Bollinger Bands
            if position == 0 and close[i] < bb_lower[i] and volume_confirm:
                position = 1
                signals[i] = position_size
            elif position == 0 and close[i] > bb_upper[i] and volume_confirm:
                position = -1
                signals[i] = -position_size
            elif position == 1 and close[i] > sma_20[i]:  # Exit at mean
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] < sma_20[i]:  # Exit at mean
                position = 0
                signals[i] = 0.0
        # In between: no clear regime, no action
    
    return signals

name = "12h_WilliamsR_1dTrend_BB_Volume"
timeframe = "12h"
leverage = 1.0