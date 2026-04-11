#!/usr/bin/env python3
"""
1d_1w_kelly_sizing_v1
Strategy: 1d Kelly sizing with weekly trend filter and volatility scaling
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses Kelly criterion to optimally size positions based on historical win rate and win/loss ratio, filtered by weekly trend direction (price above/below weekly SMA50) and scaled by volatility (inverse ATR). Designed to capture medium-term trends while dynamically adjusting position size based on edge and volatility. Works in both bull and bear markets by going long in uptrends and short in downtrends with appropriate sizing.
"""

import numpy as np
import pandas as pd
from math import sqrt
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kelly_sizing_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Daily ATR(14) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Kelly fraction based on historical performance
    # Use lookback of 60 days to estimate win rate and win/loss ratio
    lookback = 60
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    win_rate = np.zeros(n)
    win_loss_ratio = np.zeros(n)
    
    for i in range(lookback, n):
        # Get returns over lookback period
        period_returns = returns[i-lookback:i]
        # Wins are positive returns
        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate[i] = len(wins) / lookback
            avg_win = np.mean(wins)
            avg_loss = np.mean(np.abs(losses))
            win_loss_ratio[i] = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_rate[i] = 0.5  # Default if no clear wins/losses
            win_loss_ratio[i] = 1.0
    
    # Kelly fraction: f = (bp - q) / b where b = win/loss ratio, p = win rate, q = loss rate
    kelly_fraction = np.zeros(n)
    for i in range(lookback, n):
        b = win_loss_ratio[i]
        p = win_rate[i]
        q = 1 - p
        if b > 0:
            kelly_fraction[i] = (b * p - q) / b
        else:
            kelly_fraction[i] = 0
        # Cap Kelly at 0.3 to avoid overbetting
        kelly_fraction[i] = max(0, min(kelly_fraction[i], 0.3))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(kelly_fraction[i])):
            signals[i] = 0.0 if position == 0 else (kelly_fraction[i] if position == 1 else -kelly_fraction[i])
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below weekly SMA50
        uptrend = price_close > sma_50_1w_aligned[i]
        downtrend = price_close < sma_50_1w_aligned[i]
        
        # Mean reversion signal from RSI
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Volatility scaling: inverse ATR (lower volatility = higher size)
        # Normalize ATR relative to its 50-day average
        if i >= 50:
            atr_ma = np.mean(atr[max(0, i-50):i])
            vol_scale = atr_ma / atr[i] if atr[i] > 0 else 1.0
            vol_scale = max(0.5, min(vol_scale, 2.0))  # Cap scaling
        else:
            vol_scale = 1.0
        
        # Base position size from Kelly
        base_size = kelly_fraction[i]
        
        # Apply volatility scaling
        vol_scaled_size = base_size * vol_scale
        
        # Final size capped at 0.35
        final_size = min(vol_scaled_size, 0.35)
        
        # Long: RSI oversold in uptrend
        long_signal = rsi_oversold and uptrend
        
        # Short: RSI overbought in downtrend
        short_signal = rsi_overbought and downtrend
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and (rsi[i] >= 40)
        exit_short = position == -1 and (rsi[i] <= 60)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = final_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -final_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = final_size if position == 1 else (-final_size if position == -1 else 0.0)
    
    return signals