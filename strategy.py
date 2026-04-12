#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter (EMA21) and session filter (08-20 UTC)
    # Uses 4h EMA21 for trend: long only when price > EMA21, short only when price < EMA21
    # Camarilla levels from prior 4h bar: L3, H3 for mean reversion in range, H4, L4 for breakout
    # Volume confirmation: volume > 1.5 * 20-period average to filter false signals
    # Session filter: only trade 08:00-20:00 UTC to avoid low-liquidity hours
    # Discrete sizing 0.20 to minimize fee churn. Target: 15-35 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA21 for trend filter
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate prior 4h Camarilla levels (using completed 4h bar)
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    camarilla_high_4h = np.full(len(df_4h), np.nan)
    camarilla_low_4h = np.full(len(df_4h), np.nan)
    camarilla_high3_4h = np.full(len(df_4h), np.nan)
    camarilla_low3_4h = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):
        high_val = high_4h[i-1]
        low_val = low_4h[i-1]
        close_val = close_4h[i-1]
        rang = high_val - low_val
        camarilla_high_4h[i] = close_val + 1.1 * rang * 1.1 / 2  # H4
        camarilla_low_4h[i] = close_val - 1.1 * rang * 1.1 / 2   # L4
        camarilla_high3_4h[i] = close_val + 1.1 * rang * 1.1 / 4  # H3
        camarilla_low3_4h[i] = close_val - 1.1 * rang * 1.1 / 4   # L3
    
    # Align Camarilla levels to 1h (shifted by one 4h bar for completion)
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low3_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(h4_4h_aligned[i]) or 
            np.isnan(l4_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine 4h trend
        bullish_trend = close[i] > ema21_4h_aligned[i]
        bearish_trend = close[i] < ema21_4h_aligned[i]
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        # Mean reversion long: price < L3 and bullish 4h trend
        if bullish_trend:
            long_entry = (close[i] < l3_4h_aligned[i]) and volume_spike[i]
        # Mean reversion short: price > H3 and bearish 4h trend
        if bearish_trend:
            short_entry = (close[i] > h3_4h_aligned[i]) and volume_spike[i]
        # Breakout long: price > H4 and bullish 4h trend
        if bullish_trend:
            long_entry = long_entry or ((close[i] > h4_4h_aligned[i]) and volume_spike[i])
        # Breakout short: price < L4 and bearish 4h trend
        if bearish_trend:
            short_entry = short_entry or ((close[i] < l4_4h_aligned[i]) and volume_spike[i])
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = False
        short_exit = False
        
        if bullish_trend:
            long_exit = (close[i] > h3_4h_aligned[i])  # Take profit at H3
        if bearish_trend:
            short_exit = (close[i] < l3_4h_aligned[i])  # Take profit at L3
        # Trend reversal exits
        if position == 1 and bearish_trend:
            long_exit = True
        if position == -1 and bullish_trend:
            short_exit = True
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0