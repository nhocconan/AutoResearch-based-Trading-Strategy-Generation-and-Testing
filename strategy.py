#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily range breakout with 1-week trend filter and volume confirmation.
# Enters long when price breaks above prior day's high with bullish 1-week EMA and volume spike.
# Enters short when price breaks below prior day's low with bearish 1-week EMA and volume spike.
# Uses weekly EMA for trend direction to avoid counter-trend trades.
# Target: 10-25 trades/year to avoid fee drag on daily timeframe.
name = "1d_DailyBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1-week EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe (use previous week's value)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Prior day's high/low for breakout levels
    # Use shift(1) to get previous day's values (available at current day's open)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan  # First day has no previous day
    prev_low[0] = np.nan
    
    # Volume confirmation: volume > 1.8x 20-day EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above prior day's high + bullish 1w trend + volume spike
            if (price > prev_high[i] and price > ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below prior day's low + bearish 1w trend + volume spike
            elif (price < prev_low[i] and price < ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below prior day's high or trend turns bearish
            if price < prev_high[i] or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above prior day's low or trend turns bullish
            if price > prev_low[i] or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals