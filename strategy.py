#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # KAMA calculation on weekly close
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # placeholder for actual volatility sum
    # Correct ER calculation: need sum of absolute changes over period
    er = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close_1w[i] - close_1w[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
    # Smoothing constants
    sc = (er * 0.29 + 0.06) ** 2  # where 0.29 = 2/(2+1), 0.06 = 2/(30+1)
    # KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # RSI on weekly close
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Donchian breakout levels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        
        # Weekly trend filter: price vs KAMA
        price_above_kama = price_close > kama_aligned[i]
        price_below_kama = price_close < kama_aligned[i]
        
        # Weekly momentum filter: RSI extremes
        rsi_overbought = rsi_aligned[i] > 60
        rsi_oversold = rsi_aligned[i] < 40
        
        # Daily breakout conditions
        breakout_up = price_close > donchian_high[i]
        breakout_down = price_close < donchian_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above weekly KAMA (uptrend) + RSI not overbought + daily breakout up
        if price_above_kama and not rsi_overbought and breakout_up:
            enter_long = True
        
        # Short: Price below weekly KAMA (downtrend) + RSI not oversold + daily breakout down
        if price_below_kama and not rsi_oversold and breakout_down:
            enter_short = True
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = price_close < donchian_low[i] or price_close < kama_aligned[i]
        exit_short = price_close > donchian_high[i] or price_close > kama_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily trend following using weekly KAMA for trend filter, weekly RSI for momentum filter, and daily Donchian breakouts for entry.
# Enters long when: price > weekly KAMA (uptrend), RSI < 60 (not overbought), and price breaks above 20-day Donchian high.
# Enters short when: price < weekly KAMA (downtrend), RSI > 40 (not oversold), and price breaks below 20-day Donchian low.
# Exits when price breaks below Donchian low (for longs) or above Donchian high (for shorts), or when trend reverses (price crosses KAMA).
# Weekly timeframe provides robust trend/momentum filtering to avoid whipsaws, while daily breakouts capture timely entries.
# Designed to work in both bull and bear markets by following the weekly trend and using momentum filters to avoid extremes.
# Position size 0.25 limits risk, and strict entry conditions aim for 10-25 trades per year to minimize fee drag.