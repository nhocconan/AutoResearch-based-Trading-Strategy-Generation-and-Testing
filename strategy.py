#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_Strategy
Hypothesis: Adaptive Kelly sizing based on 60-period win rate and volatility (ATR) improves risk-adjusted returns.
Uses 60-bar lookback to estimate win probability and average win/loss, then applies Kelly fraction scaled by volatility.
Incorporates 12h trend filter (EMA50) to avoid counter-trend trades. Targets 15-25 trades/year on 6h.
Works in bull/bear via trend filter and volatility-adjusted sizing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate returns for Kelly calculation
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    lookback = 60  # 60-period lookback for Kelly
    
    # Warmup period
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        trend = ema50_12h_aligned[i]
        
        if i >= lookback:
            # Calculate win rate and avg win/loss over lookback period
            period_returns = returns[i-lookback+1:i+1]
            winning_returns = period_returns[period_returns > 0]
            losing_returns = period_returns[period_returns < 0]
            
            win_rate = len(winning_returns) / lookback if lookback > 0 else 0.5
            avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
            avg_loss = np.mean(np.abs(losing_returns)) if len(losing_returns) > 0 else 0
            
            # Kelly fraction: f = (bp - q) / b where b = avg_win/avg_loss, p = win_rate, q = loss_rate
            if avg_loss > 0 and avg_win > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - p
                kelly = (b * p - q) / b if b > 0 else 0
                kelly = max(0, min(kelly, 0.5))  # Cap at 0.5 (half-Kelly)
            else:
                kelly = 0
            
            # Volatility scaling: reduce size in high volatility
            vol_factor = 1 / (1 + atr[i] * 100)  # Normalize ATR influence
            size = kelly * vol_factor * 0.5  # Base scaling
            
            # Ensure minimum size and cap
            size = max(0.05, min(size, 0.30))
            
            # Entry logic: trend following with volatility filter
            if position == 0:
                # Long in uptrend with moderate volatility
                if close[i] > trend and atr[i] < np.percentile(atr[max(0, i-50):i+1], 80):
                    signals[i] = size
                    position = 1
                # Short in downtrend with moderate volatility
                elif close[i] < trend and atr[i] < np.percentile(atr[max(0, i-50):i+1], 80):
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long: trend reversal or volatility spike
                if close[i] < trend or atr[i] > np.percentile(atr[max(0, i-20):i+1], 90):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif position == -1:
                # Exit short: trend reversal or volatility spike
                if close[i] > trend or atr[i] > np.percentile(atr[max(0, i-20):i+1], 90):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Adaptive_Kelly_Strategy"
timeframe = "6h"
leverage = 1.0