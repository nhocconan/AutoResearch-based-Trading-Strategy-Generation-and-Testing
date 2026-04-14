#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 1-day Donchian breakout and volume confirmation
# Choppiness Index (CHOP) > 61.8 = ranging market (mean revert at Donchian bands)
# CHOP < 38.2 = trending market (breakout in direction of trend)
# In ranging markets: long at lower Donchian band, short at upper band
# In trending markets: long on breakout above upper band, short on breakdown below lower band
# Volume confirmation required: volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel or CHOP regime shifts
# This adapts to both trending and ranging markets, reducing false breakouts in low volatility
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily Donchian channels for trend direction (55-period)
    high_55_1d = pd.Series(df_1d['high'].values).rolling(window=55, min_periods=55).max().values
    low_55_1d = pd.Series(df_1d['low'].values).rolling(window=55, min_periods=55).min().values
    high_55_1d_aligned = align_htf_to_ltf(prices, df_1d, high_55_1d)
    low_55_1d_aligned = align_htf_to_ltf(prices, df_1d, low_55_1d)
    
    # Calculate Choppiness Index on 4h (14-period)
    # CHOP = 100 * log10(sum(ATR over n) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((range_hl == 0) | np.isnan(atr_sum) | np.isnan(range_hl), 50.0, chop)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20, 55, 14)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Determine market regime based on Choppiness Index
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            if is_ranging:
                # In ranging markets: mean revert at Donchian bands
                if price <= low_20[i] and vol > vol_threshold:
                    position = 1
                    signals[i] = position_size
                elif price >= high_20[i] and vol > vol_threshold:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif is_trending:
                # In trending markets: breakout in direction of trend
                # Use daily Donchian to determine trend direction
                trend_up = price > high_55_1d_aligned[i]
                trend_down = price < low_55_1d_aligned[i]
                
                if trend_up and price > high_20[i] and vol > vol_threshold:
                    position = 1
                    signals[i] = position_size
                elif trend_down and price < low_20[i] and vol > vol_threshold:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # In transition zone (38.2 <= CHOP <= 61.8): no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back above lower Donchian band OR regime shifts to strong trending against position
            if price >= low_20[i] or (is_trending and not trend_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back below upper Donchian band OR regime shifts to strong trending against position
            if price <= high_20[i] or (is_trending and not trend_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_Regime_Volume"
timeframe = "4h"
leverage = 1.0