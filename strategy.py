#!/usr/bin/env python3
"""
exp_7494_1h_4h_1d_rsi_trend_v1
Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d momentum filter. 
In bull markets (price > 1d EMA200): buy RSI<30 dips in uptrend. 
In bear markets (price < 1d EMA200): sell RSI>70 rallies in downtrend. 
Uses 4h RSI for trend strength to avoid chop. 
Targets 60-150 trades over 4 years (15-37/year) with strict RSI extremes + trend alignment.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7494_1h_4h_1d_rsi_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_FAST = 20
EMA_SLOW = 50
EMA_TREND = 200
RSI_TREND = 14
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h RSI for trend filter
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/RSI_TREND, adjust=False, min_periods=RSI_TREND).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/RSI_TREND, adjust=False, min_periods=RSI_TREND).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d EMA200 for regime filter
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI for entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMAs for trend confirmation
    ema_fast = pd.Series(close).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_SLOW, EMA_TREND, RSI_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_1d_200_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        above_ema200 = close[i] > ema_1d_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_1d_200_aligned[i]  # bear regime
        strong_uptrend_4h = rsi_4h_aligned[i] > 50       # 4h bullish
        strong_downtrend_4h = rsi_4h_aligned[i] < 50     # 4h bearish
        
        # EMA alignment for micro-trend
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # Entry conditions
        long_entry = (
            above_ema200 and           # bull regime
            strong_uptrend_4h and      # 4h bullish
            ema_bullish and            # 1h EMA aligned
            rsi[i] < RSI_OVERSOLD      # oversold
        )
        
        short_entry = (
            below_ema200 and           # bear regime
            strong_downtrend_4h and    # 4h bearish
            ema_bearish and            # 1h EMA aligned
            rsi[i] > RSI_OVERBOUGHT    # overbought
        )
        
        # Exit conditions
        long_exit = rsi[i] > 50  # exit long when RSI returns to neutral
        short_exit = rsi[i] < 50  # exit short when RSI returns to neutral
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals