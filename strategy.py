#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + chop regime filter
# - Long: KAMA rising (trend up) + RSI < 40 (pullback) + CHOP > 61.8 (range/transition)
# - Short: KAMA falling (trend down) + RSI > 60 (bounce) + CHOP > 61.8 (range/transition)
# - Exit: Opposite RSI extreme (RSI > 60 for long exit, RSI < 40 for short exit) or KAMA trend reversal
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-80 trades/year (80-320 total over 4 years) to balance opportunity and fees
# - Works in both bull/bear: KAMA captures trend, RSI catches mean reversion in chop, CHOP filter avoids strong trends where mean reversion fails

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w ATR for chop regime calculation
    tr_1w = np.maximum(df_1w['high'] - df_1w['low'], 
                       np.maximum(np.abs(df_1w['high'] - np.roll(df_1w['close'], 1)), 
                                  np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))))
    tr_1w[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # True Range for chop calculation (need full period range)
    # CHOP = 100 * log10(sum(ATR14) / (max(highN) - min(lowN))) / log10(N)
    # We'll use 14-period chop as standard
    highest_high_1w = df_1w['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_1w = df_1w['low'].rolling(window=14, min_periods=14).min().values
    sum_atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    range_1w = highest_high_1w - lowest_low_1w
    chop_1w = 100 * np.log10(sum_atr_1w / range_1w) / np.log10(14)
    chop_1w = np.where(range_1w > 0, chop_1w, 50)  # avoid div/0, set to neutral when range=0
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Pre-compute KAMA on daily close
    # KAMA: ER = |Close - Close[10]| / Sum(|Close - Close[1]|, 10)
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA[i] = KAMA[i-1] + SC * (Price[i] - KAMA[i-1])
    close_series = pd.Series(close)
    change = np.abs(close_series - close_series.shift(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = np.where(volatility > 0, er, 0)
    fastest_sc = 2 / (2 + 1)  # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute RSI(14)
    delta = close_series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # all gains -> RSI=100
    rsi = np.where(avg_gain == 0, 0, rsi)   # all losses -> RSI=0
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction (using 2-period change to avoid noise)
        kama_rising = kama[i] > kama[i-2]
        kama_falling = kama[i] < kama[i-2]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Chop regime: CHOP > 61.8 indicates ranging/transition market (good for mean reversion)
        chop_high = chop_1w_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA up (trend) + RSI oversold (pullback) + choppy market (mean reversion favorable)
        if kama_rising and rsi_oversold and chop_high:
            enter_long = True
        
        # Short: KAMA down (trend) + RSI overbought (bounce) + choppy market (mean reversion favorable)
        if kama_falling and rsi_overbought and chop_high:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: RSI overbought (mean reversion complete) OR KAMA trend turns down
            exit_long = rsi_overbought or kama_falling
        elif position == -1:
            # Exit short: RSI oversold (mean reversion complete) OR KAMA trend turns up
            exit_short = rsi_oversold or kama_rising
        
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