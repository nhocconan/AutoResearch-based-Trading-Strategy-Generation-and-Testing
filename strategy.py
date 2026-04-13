#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
    # 4h EMA50 determines trend: price above EMA50 = bullish bias (long breakouts only),
    # price below EMA50 = bearish bias (short breakouts only). Camarilla pivot levels (H3/L3)
    # from 1h data provide precise entry/exit levels. Volume confirmation ensures breakout validity.
    # Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot levels (H3, L3, H4, L4)
    # Camarilla levels based on prior period's range
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    high_shift = pd.Series(high).shift(1)
    low_shift = pd.Series(low).shift(1)
    close_shift = pd.Series(close).shift(1)
    
    # Prior period's range
    range_hl = high_shift - low_shift
    
    # Camarilla levels
    H3 = close_shift + 1.1 * range_hl / 4
    L3 = close_shift - 1.1 * range_hl / 4
    H4 = close_shift + 1.1 * range_hl / 2
    L4 = close_shift - 1.1 * range_hl / 2
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(H4[i]) or np.isnan(L4[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        volume_filter = volume[i] > volume_ma
        
        # Trend bias from 4h EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        # Breakout conditions using Camarilla levels
        long_breakout = close[i] > H3[i] and close[i] < H4[i]  # Break above H3 but below H4 (avoid false breakouts)
        short_breakout = close[i] < L3[i] and close[i] > L4[i]  # Break below L3 but above L4
        
        # Entry conditions: breakout in direction of 4h trend
        long_entry = long_breakout and bullish_bias and volume_filter
        short_entry = short_breakout and bearish_bias and volume_filter
        
        # Exit conditions: opposite breakout or loss of trend bias
        long_exit = short_breakout or not bullish_bias
        short_exit = long_breakout or not bearish_bias
        
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

name = "1h_4h_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0