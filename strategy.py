#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction (trend filter) and 1h RSI(2) pullback for entry timing.
# Long when price breaks above 4h Donchian upper(20) AND 1h RSI(2) < 10 (deep pullback in uptrend).
# Short when price breaks below 4h Donchian lower(20) AND 1h RSI(2) > 90 (overbought retracement in downtrend).
# Exit when price crosses 4h Donchian midpoint (mean reversion) OR RSI(2) reaches opposite extreme.
# Uses discrete sizing 0.20 to minimize fee churn and manage drawdown.
# Target: 60-120 total trades over 4 years (15-30/year) for 1h timeframe.
# Works in bull markets (buying pullbacks in uptrends) and bear markets (selling retracements in downtrends).
# Session filter 08-20 UTC reduces noise and focuses on liquid hours.

name = "1h_Donchian20_4hTrend_RSI2_Pullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least one completed 4h bar for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align 4h Donchian to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Calculate 1h RSI(2) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND 1h RSI(2) < 10 (deep pullback)
            if close[i] > donchian_upper_aligned[i] and rsi[i] < 10:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower AND 1h RSI(2) > 90 (overbought retracement)
            elif close[i] < donchian_lower_aligned[i] and rsi[i] > 90:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses 4h Donchian midpoint OR RSI(2) > 90 (overbought)
            if close[i] < donchian_mid_aligned[i] or rsi[i] > 90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses 4h Donchian midpoint OR RSI(2) < 10 (oversold)
            if close[i] > donchian_mid_aligned[i] or rsi[i] < 10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals