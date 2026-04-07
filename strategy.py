#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour price action respecting 4-hour and daily support/resistance
# - Use 4h Donchian channels (20-period) for trend direction
# - Use daily high/low for key support/resistance levels
# - Enter long when price bounces off daily low with 4h uptrend
# - Enter short when price rejects daily high with 4h downtrend
# - Volume confirmation to filter noise
# - Session filter (08-20 UTC) to avoid low liquidity periods
# - Target: 15-37 trades per year (60-150 over 4 years) to minimize fee drag
# - Works in bull/bear markets by using multiple timeframe confluence

name = "1h_donchian4h_daily_levels_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Get 4h data for Donchian channels (trend direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get daily data for key support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily high and low as key levels
    daily_high = high_1d
    daily_low = low_1d
    
    # Align daily levels to 1h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or
            np.isnan(vol_ma[i]) or hours[i] < 8 or hours[i] > 20):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend direction from 4h Donchian
        # Uptrend: price above Donchian middle, Downtrend: price below Donchian middle
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        uptrend_4h = close_4h[-1] > donchian_mid if len(close_4h) > 0 else False  # Simplified - use aligned value
        
        # Use current bar's close for trend determination via aligned 4h data
        # Find the 4h bar that corresponds to current 1h bar
        # Since we have aligned arrays, we can infer trend from price position relative to Donchian
        # Simple approach: if current close > Donchian high, uptrend; if < Donchian low, downtrend
        if close[i] > donchian_high_aligned[i]:
            trend_4h = 1  # Uptrend
        elif close[i] < donchian_low_aligned[i]:
            trend_4h = -1  # Downtrend
        else:
            trend_4h = 0  # Ranging
        
        # Price levels
        dh = daily_high_aligned[i]
        dl = daily_low_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below daily low (support break) or reaches daily high (take profit)
            if close[i] < dl or close[i] > dh:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above daily high (resistance break) or reaches daily low (take profit)
            if close[i] > dh or close[i] < dl:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Long: bounce off daily low with 4h uptrend bias or break above daily high with volume
            # Short: rejection at daily high with 4h downtrend bias or break below daily low with volume
            
            # Long conditions
            long_condition = False
            # Bounce off daily low with some tolerance and 4h not strongly downtrend
            if abs(close[i] - dl) < 0.002 * dl and trend_4h != -1 and vol_confirm:
                long_condition = True
            # Break above daily high with volume
            elif close[i] > dh and vol_confirm:
                long_condition = True
            
            # Short conditions
            short_condition = False
            # Rejection at daily high with some tolerance and 4h not strongly uptrend
            if abs(close[i] - dh) < 0.002 * dh and trend_4h != 1 and vol_confirm:
                short_condition = True
            # Break below daily low with volume
            elif close[i] < dl and vol_confirm:
                short_condition = True
            
            if long_condition:
                position = 1
                signals[i] = 0.20
            elif short_condition:
                position = -1
                signals[i] = -0.20
    
    return signals