#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h Donchian channel (20) for trend direction
# - 1h Camarilla levels (H3/L3) for breakout entries
# - Volume confirmation: current volume > 1.5x 20-period average
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work well in ranging markets; Donchian filter ensures we only trade in trends
# - Volume confirmation filters out false breakouts
# - Session filter reduces noise during Asian session

name = "1h_4h_camarilla_donchian_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # Pre-compute 1h Camarilla levels (based on previous day's OHLC)
    # We'll use daily OHLC from 1d timeframe for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1h bar based on prior day's OHLC
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # But we need to align daily OHLC to 1h bars
    d_close = df_1d['close'].values
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    
    # True range for Camarilla calculation
    tr = d_high - d_low
    
    # Camarilla levels (using previous day's values)
    camarilla_h4 = d_close + 1.5 * tr
    camarilla_h3 = d_close + 1.1 * tr
    camarilla_l3 = d_close - 1.1 * tr
    camarilla_l4 = d_close - 1.5 * tr
    
    # Align daily Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Current price data
        close_current = close[i]
        high_current = high[i]
        low_current = low[i]
        volume_current = volume[i]
        
        # Trend filter: price above/below 4h Donchian
        uptrend = close_current > donchian_high_aligned[i]
        downtrend = close_current < donchian_low_aligned[i]
        
        # Camarilla breakout conditions
        breakout_long = close_current > h3_aligned[i]  # Break above H3
        breakout_short = close_current < l3_aligned[i]  # Break below L3
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: uptrend + breakout above H3 + volume confirmation
        if uptrend and breakout_long and vol_confirm:
            enter_long = True
        
        # Short: downtrend + breakout below L3 + volume confirmation
        if downtrend and breakout_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Donchian break or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below Donchian low OR trend turns down
            exit_long = (close_current < donchian_low_aligned[i]) or (not uptrend)
        elif position == -1:
            # Exit short if price breaks above Donchian high OR trend turns up
            exit_short = (close_current > donchian_high_aligned[i]) or (not downtrend)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals