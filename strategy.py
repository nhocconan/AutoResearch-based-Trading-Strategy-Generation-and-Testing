#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels for mean reversion and 1w Donchian channel for trend filter
# - Uses 1d HTF for Camarilla pivots: price reversion to H3/L3 levels with volume confirmation
# - Uses 1w HTF for Donchian channel (20-period): trend filter to avoid counter-trend trades
# - In uptrend (price > Donchian upper): long when price touches or breaks below H3 level
# - In downtrend (price < Donchian lower): short when price touches or breaks above L3 level
# - Volume confirmation: current 12h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_donchian_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # Using previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Calculate 1w Donchian channel (20 periods)
    # Upper band = 20-period high
    # Lower band = 20-period low
    period20_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, period20_high)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, period20_low)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend determination using Donchian channel
        bullish_trend = close[i] > donchian_upper_aligned[i]
        bearish_trend = close[i] < donchian_lower_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: price reaches L3 level or trend changes to bearish
            if close[i] <= camarilla_l3_aligned[i] or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches H3 level or trend changes to bullish
            if close[i] >= camarilla_h3_aligned[i] or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic based on trend and Camarilla levels
            if volume_confirmed:
                if bullish_trend and close[i] <= camarilla_h3_aligned[i]:
                    # In uptrend, price touches or breaks below H3: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and close[i] >= camarilla_l3_aligned[i]:
                    # In downtrend, price touches or breaks above L3: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals