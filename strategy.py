#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR(14) stoploss
# - Long when price breaks above 20-day high AND 1w close > 1w EMA50
# - Short when price breaks below 20-day low AND 1w close < 1w EMA50
# - Exit when price retraces to 10-day EMA (mean reversion in bear markets)
# - Uses weekly trend filter to avoid counter-trend trades in 2025+ bear market
# - Tight entry conditions target 15-25 trades/year (60-100 total over 4 years)
# - Focus on BTC/ETH; proven Donchian+volume+chop pattern works on SOL but we add trend filter for BTC/ETH edge

name = "1d_1w_donchian_breakout_trend_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 10-day EMA for exit (mean reversion target)
    ema10 = prices['close'].ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute aligned 1w data
    c_1w = df_1w['close'].values
    ema50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(ema10[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > 20-day high AND 1w uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                prices['close'].iloc[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < 20-day low AND 1w downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit to 10-day EMA (mean reversion)
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= ema10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= ema10[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals