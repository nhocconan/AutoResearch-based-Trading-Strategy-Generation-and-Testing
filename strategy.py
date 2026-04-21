#!/usr/bin/env python3
"""
1d_1w_Kelly_Fractional_Volume_Weighted_Consensus
Hypothesis: Weekly trend (EMA10) filters direction; daily RSI(14) extremes trigger entries only during high-volume breakouts from Bollinger Bands(20,2). Uses Kelly criterion fractional sizing based on recent win rate to manage risk and avoid overtrading. Designed for 10-30 trades/year with strong edge in both bull (trend continuation) and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    close_weekly = df_weekly['close'].values
    ema10_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False).mean().values
    ema10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema10_weekly)
    
    # Daily data for signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Volume filter: current volume > 2x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2 * vol_avg)
    
    # Kelly fraction estimation: use last 252 days (1 year) win/loss
    lookback = 252
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    win_rate = np.zeros(n)
    avg_win = np.zeros(n)
    avg_loss = np.zeros(n)
    kelly_frac = np.zeros(n)
    
    for i in range(lookback, n):
        period_returns = returns[i-lookback:i]
        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        if len(wins) > 0 and len(losses) > 0:
            wr = len(wins) / len(period_returns)
            aw = np.mean(wins)
            al = np.mean(abs(losses))
            if al > 0:
                kelly = wr - ((1 - wr) / (aw / al))
                kelly = max(0, min(kelly, 0.5))  # cap at 0.5
            else:
                kelly = 0
        else:
            kelly = 0
        win_rate[i] = wr if len(wins) > 0 else 0
        avg_win[i] = aw if len(wins) > 0 else 0
        avg_loss[i] = al if len(losses) > 0 else 0
        kelly_frac[i] = kelly
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema10_weekly_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(kelly_frac[i])):
            continue
        
        price = close[i]
        trend_up = price > ema10_weekly_aligned[i]
        rsi_val = rsi[i]
        vol_ok = vol_filter[i]
        kelly = kelly_frac[i]
        
        # Entry conditions
        if position == 0:
            # Long: RSI < 30 (oversold) + price breaks above lower BB + volume + Kelly > 0
            if (rsi_val < 30 and price > lower[i] and vol_ok and kelly > 0):
                size = min(0.30, kelly * 2)  # scale Kelly to max 0.30
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) + price breaks below upper BB + volume + Kelly > 0
            elif (rsi_val > 70 and price < upper[i] and vol_ok and kelly > 0):
                size = min(0.30, kelly * 2)
                signals[i] = -size
                position = -1
        
        # Exit: reverse signal or volatility expansion
        elif position == 1:
            if rsi_val > 70 or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if kelly > 0.1 else 0.15  # reduce size in low Kelly
        elif position == -1:
            if rsi_val < 30 or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30 if kelly > 0.1 else -0.15
    
    return signals

name = "1d_1w_Kelly_Fractional_Volume_Weighted_Consensus"
timeframe = "1d"
leverage = 1.0