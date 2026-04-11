#!/usr/bin/env python3
# 1d_1w_kelly_volatility_v1
# Strategy: 1d Kelly fraction position sizing based on volatility-adjusted momentum with 1w trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Kelly criterion optimizes position sizing based on win probability and payoff ratio.
# Uses 1w EMA for trend direction and daily volatility for position sizing. Low frequency (~10-20/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kelly_volatility_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily returns for win probability estimation
    returns = np.diff(close, prepend=close[0]) / close
    
    # Rolling win probability and average win/loss over 60 days
    win_prob = np.zeros(n)
    avg_win = np.zeros(n)
    avg_loss = np.zeros(n)
    
    for i in range(60, n):
        period_returns = returns[i-60:i]
        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        
        if len(wins) > 0:
            avg_win[i] = np.mean(wins)
        if len(losses) > 0:
            avg_loss[i] = np.mean(-losses)  # positive value
        
        if len(period_returns) > 0:
            win_prob[i] = len(wins) / len(period_returns)
    
    # Kelly fraction: f = (bp - q) / b where b = avg_win/avg_loss, p = win_prob, q = 1-p
    kelly_fraction = np.zeros(n)
    for i in range(60, n):
        if avg_loss[i] > 0 and avg_win[i] > 0:
            b = avg_win[i] / avg_loss[i]
            p = win_prob[i]
            q = 1 - p
            kelly = (b * p - q) / b
            # Cap Kelly at 0.3 and floor at 0
            kelly_fraction[i] = max(0, min(0.3, kelly))
    
    # Volatility adjustment: reduce size in high volatility
    # Calculate 10-day ATR as volatility proxy
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Normalize ATR by price for volatility regime
    atr_ratio = np.zeros(n)
    for i in range(10, n):
        if close[i] > 0:
            atr_ratio[i] = atr_10[i] / close[i]
    
    # Volatility regime: reduce size when volatility is high (above 70th percentile)
    vol_threshold = np.zeros(n)
    for i in range(60, n):
        if i >= 60:
            vol_threshold[i] = np.percentile(atr_ratio[max(0, i-60):i], 70)
    
    vol_adjustment = np.ones(n)
    for i in range(60, n):
        if atr_ratio[i] > vol_threshold[i]:
            vol_adjustment[i] = 0.5  # Reduce size by half in high volatility
    
    # Final position size: Kelly * volatility adjustment
    position_size = kelly_fraction * vol_adjustment
    
    # Trend filter: only take positions aligned with 1w trend
    uptrend = close > ema_21_1w_aligned
    downtrend = close < ema_21_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(position_size[i]):
            signals[i] = 0.0 if position == 0 else (position_size[i] if position == 1 else -position_size[i])
            continue
        
        # Entry logic: Kelly sizing with trend alignment
        if uptrend[i] and position != 1:
            position = 1
            signals[i] = position_size[i]
        elif downtrend[i] and position != -1:
            position = -1
            signals[i] = -position_size[i]
        # Exit: trend reversal
        elif position == 1 and not uptrend[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not downtrend[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size[i]
            elif position == -1:
                signals[i] = -position_size[i]
            else:
                signals[i] = 0.0
    
    return signals