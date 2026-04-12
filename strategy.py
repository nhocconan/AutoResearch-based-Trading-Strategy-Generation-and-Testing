#1. State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 1d_1w_donchian_breakout_v2
# Weekly Donchian channel breakout with volume confirmation on daily chart.
# In bull markets: buy breakouts above weekly high with volume surge.
# In bear markets: sell breakdowns below weekly low with volume surge.
# Uses weekly trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol for low friction and high win rate.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (waits for weekly close)
    donchian_high = align_htf_to_ltf(prices, df_1w, weekly_high)
    donchian_low = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: volume > 2.0 * 20-period average on daily
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # Weekly trend filter: price above/below weekly 50-period SMA
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    uptrend = close > weekly_sma50_aligned
    downtrend = close < weekly_sma50_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly Donchian high with volume and uptrend
        if close[i] > donchian_high[i] and vol_confirm[i] and uptrend[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly Donchian low with volume and downtrend
        elif close[i] < donchian_low[i] and vol_confirm[i] and downtrend[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or trend reversal
        elif (close[i] < donchian_low[i] and position == 1) or \
             (close[i] > donchian_high[i] and position == -1) or \
             (position == 1 and not uptrend[i]) or \
             (position == -1 and not downtrend[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals