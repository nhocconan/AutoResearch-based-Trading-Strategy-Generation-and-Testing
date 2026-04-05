#!/usr/bin/env python3
"""
Experiment #7874: 1-hour RSI mean reversion with 4h trend filter and session filter.
Hypothesis: In ranging markets (common in 2025), RSI extremes combined with 4h trend direction provide high-probability mean reversion trades. The 4h trend filter ensures trades align with higher timeframe momentum, while the session filter (08-20 UTC) avoids low-liquidity periods. Targets 60-150 trades over 4 years with controlled position sizing.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7874_1h_rsi_meanrev_4h_trend_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SIGNAL_SIZE = 0.20
EMA_PERIOD_4H = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD_4H, adjust=False, min_periods=EMA_PERIOD_4H).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    avg_loss = pd.Series(loss).ewm(span=RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_PERIOD_4H, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_bias_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + (ATR_STOP_MULTIPLIER * atr[i]):
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h EMA
        bull_bias = trend_bias_4h_aligned[i] == 1   # 4h close above EMA
        bear_bias = trend_bias_4h_aligned[i] == -1  # 4h close below EMA
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry conditions
        long_entry = bull_bias and rsi_oversold
        short_entry = bear_bias and rsi_overbought
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals