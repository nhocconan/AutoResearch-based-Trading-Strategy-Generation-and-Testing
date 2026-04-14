#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy with 1w Supertrend trend filter and 1d ATR-based volatility breakout
# Supertrend (10, 3) on weekly timeframe determines trend direction
# Volatility breakout: price closes above/below ATR-based bands (mean ± k*ATR) with volume confirmation
# Works in both bull and bear markets: Supertrend filter ensures we trade with the higher timeframe trend,
# while volatility breakout captures momentum expansion phases. Volume filter avoids false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for Supertrend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Supertrend (10, 3)
    atr_len = 10
    atr_mult = 3.0
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1w, np.nan)
    direction = np.full_like(close_1w, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if np.isnan(atr[i-1]) or np.isnan(close_1w[i-1]):
            supertrend[i] = np.nan
            direction[i] = 1
            continue
            
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 1d timeframe
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d ATR for volatility breakout bands
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_d = pd.Series(tr_d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d SMA (20) for mean
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volatility bands: mean ± k*ATR
    k = 1.5
    upper_vol_band = sma_20 + (k * atr_d)
    lower_vol_band = sma_20 - (k * atr_d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Supertrend needs ~10, SMA needs 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(direction_aligned[i]) or 
            np.isnan(atr_d[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: Supertrend direction from weekly
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Volatility breakout conditions
        breakout_up = price > upper_vol_band[i]
        breakout_down = price < lower_vol_band[i]
        
        if position == 0:
            # Enter long: uptrend + upward breakout + volume
            if uptrend and breakout_up and vol_filter[i]:
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + downward breakout + volume
            elif downtrend and breakout_down and vol_filter[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below SMA (mean reversion) OR trend changes
            if price < sma_20[i] or direction_aligned[i] != 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above SMA (mean reversion) OR trend changes
            if price > sma_20[i] or direction_aligned[i] != -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wSupertrend_ATRBreakout_Volume_v1"
timeframe = "1d"
leverage = 1.0