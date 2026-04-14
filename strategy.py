#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 12h EMA trend and volume confirmation
# Uses Choppiness Index (14-period) on 4h to detect ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets
# In trending markets (CHOP < 38.2): follow 12h EMA direction (price > EMA = long, price < EMA = short)
# In ranging markets (CHOP > 61.8): mean revert at Bollinger Bands (2, 2 std) - sell at upper band, buy at lower band
# Volume confirmation: require volume > 1.2x 20-period EMA to avoid false signals
# Designed for ~20-30 trades/year with regime adaptation for both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Choppiness Index (14-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (14-period)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of TR over 14 periods
    sum_tr = tr.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values  # Handle division by zero
    
    # Calculate EMA (21-period) on 12h for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bollinger Bands (20, 2) on 4h for mean reversion in ranging markets
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
        # Get aligned 12h EMA
        ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)[i]
        
        if np.isnan(chop[i]) or np.isnan(ema_12h_aligned) or np.isnan(vol_ma[i]) or \
           np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # Volume confirmation (1.2x average)
        volume_confirm = volume[i] > 1.2 * vol_ma[i]
        
        # Regime-based logic
        if chop[i] < 38.2:  # Trending market - follow 12h EMA
            if position == 0 and close[i] > ema_12h_aligned and volume_confirm:
                position = 1
                signals[i] = position_size
            elif position == 0 and close[i] < ema_12h_aligned and volume_confirm:
                position = -1
                signals[i] = -position_size
            elif position == 1 and close[i] < ema_12h_aligned:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > ema_12h_aligned:
                position = 0
                signals[i] = 0.0
        elif chop[i] > 61.8:  # Ranging market - mean revert at Bollinger Bands
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
        # In between (38.2 <= CHOP <= 61.8): neutral, no action
    
    return signals

name = "4h_Chop_Regime_EMA_BB_Volume"
timeframe = "4h"
leverage = 1.0