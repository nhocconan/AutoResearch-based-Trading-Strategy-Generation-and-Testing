#!/usr/bin/env python3
"""
1d_1w_RSI_Pullback_with_Volume_Confirmation
Hypothesis: On daily timeframe, enter long when RSI(14) crosses above 30 from below (bullish momentum in uptrend)
with volume > 1.5x 20-day average, and price above 200-day EMA (trend filter). Enter short when RSI crosses below 70
from above with volume expansion and price below 200-day EMA. Uses weekly trend filter: only take longs when
weekly close > weekly 50 EMA, shorts when weekly close < weekly 50 EMA. Designed for 1d timeframe to target
15-25 trades/year (60-100 total over 4 years). Works in bull markets via RSI pullback longs and in bear markets
via RSI rejection shorts, both requiring volume confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily indicators
    close_series = pd.Series(close)
    rsi_period = 14
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    ema200 = close_series.ewm(span=200, min_periods=200).mean().values
    
    vol_ma_20 = close_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(x), raw=True).values
    # Actually compute volume MA correctly
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(rsi_values[i]) or 
            np.isnan(ema200[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        volume_expansion = volume[i] > (vol_ma_20[i] * 1.5)
        
        # RSI crossover conditions
        rsi_now = rsi_values[i]
        rsi_prev = rsi_values[i-1]
        rsi_long_signal = (rsi_now > 30) and (rsi_prev <= 30)  # Cross above 30
        rsi_short_signal = (rsi_now < 70) and (rsi_prev >= 70)  # Cross below 70
        
        # Trend filters
        price_above_ema200 = close[i] > ema200[i]
        price_below_ema200 = close[i] < ema200[i]
        weekly_uptrend = weekly_ema50_aligned[i] > weekly_ema50_aligned[i-1]  # Weekly EMA50 rising
        weekly_downtrend = weekly_ema50_aligned[i] < weekly_ema50_aligned[i-1]  # Weekly EMA50 falling
        
        # Entry conditions
        long_entry = rsi_long_signal and volume_expansion and price_above_ema200 and weekly_uptrend
        short_entry = rsi_short_signal and volume_expansion and price_below_ema200 and weekly_downtrend
        
        # Exit conditions: reverse signal or RSI midpoint crossover
        exit_long = position == 1 and (rsi_now < 50 or rsi_short_signal)
        exit_short = position == -1 and (rsi_now > 50 or rsi_long_signal)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_RSI_Pullback_with_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0