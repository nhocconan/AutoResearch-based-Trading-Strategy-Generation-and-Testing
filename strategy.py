#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily ATR Expansion with Volume and Trend Filter
# Hypothesis: In both bull and bear markets, significant volatility expansion (ATR spike)
# combined with volume surge and trend alignment (price vs 50 EMA) captures strong
# directional moves. Enter long on bullish expansion (close > open + ATR expansion),
# short on bearish expansion (close < open - ATR expansion). Exit when volatility
# contracts or trend weakens. Designed for 6h timeframe with 1d ATR for stability.
# Target: 12-25 trades/year (48-100 over 4 years).

name = "6h_daily_atr_expansion_volume_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation (more stable on 6s timeframe)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily data
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_d = np.zeros_like(tr)
    atr_d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_d[i] = (atr_d[i-1] * 13 + tr[i]) / 14
    
    # Shift ATR by 1 to use previous day's value (avoid look-ahead)
    atr_d = np.roll(atr_d, 1)
    if len(atr_d) > 1:
        atr_d[0] = atr_d[1]
    else:
        atr_d[0] = 0.0
    
    # Align ATR to 6h timeframe
    atr_d_aligned = align_htf_to_ltf(prices, df_daily, atr_d)
    
    # Trend filter: price vs 50 EMA on 6s timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(atr_d_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate 6s bar range and body
        bar_range = high[i] - low[i]
        if bar_range == 0:
            signals[i] = 0.0
            continue
            
        body_size = abs(close[i] - open_prices[i])
        is_bullish = close[i] > open_prices[i]
        is_bearish = close[i] < open_prices[i]
        
        # ATR expansion condition: current bar range > 1.5x daily ATR
        atr_expansion = bar_range > (1.5 * atr_d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: trend failure or volatility contraction
            if close[i] <= ema_50[i] or not atr_expansion or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: trend failure or volatility contraction
            if close[i] >= ema_50[i] or not atr_expansion or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: bullish bar with ATR expansion, above EMA, volume surge
            if (is_bullish and atr_expansion and 
                close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish bar with ATR expansion, below EMA, volume surge
            elif (is_bearish and atr_expansion and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals