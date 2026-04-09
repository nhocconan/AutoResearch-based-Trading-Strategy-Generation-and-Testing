#!/usr/bin/env python3
# 1h_4h1d_trend_ema_rsi_v1
# Hypothesis: 1h strategy using 4h EMA trend and 1d RSI for regime filter, with 1h EMA pullback entries.
# Long: 4h EMA21 up, 1d RSI>50, 1h price pulls back to EMA21 with RSI<30.
# Short: 4h EMA21 down, 1d RSI<50, 1h price pulls back to EMA21 with RSI>70.
# Exit: Opposite EMA21 cross or RSI extreme reversal.
# Uses 4h for trend direction, 1d for market regime, 1h for precise entry timing.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_ema_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h EMA21 for entry timing
    close_s = pd.Series(close)
    ema21_1h = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # 1h RSI14 for entry timing
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14_1h = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Get 4h data for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # 4h EMA21 for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema21_4h = close_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # 1d RSI14 for regime
    close_1d = pd.Series(df_1d['close'].values)
    delta_1d = close_1d.diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi14_1d = (100 - (100 / (1 + rs_1d))).fillna(50).values
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or
            np.isnan(ema21_1h[i]) or np.isnan(rsi14_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend and regime
        uptrend_4h = close[i] > ema21_4h_aligned[i]
        downtrend_4h = close[i] < ema21_4h_aligned[i]
        bullish_regime = rsi14_1d_aligned[i] > 50
        bearish_regime = rsi14_1d_aligned[i] < 50
        
        if position == 1:  # Long position
            # Exit: Price breaks below EMA21 or RSI exceeds 70 (overbought)
            if close[i] < ema21_1h[i] or rsi14_1h[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price breaks above EMA21 or RSI falls below 30 (oversold)
            if close[i] > ema21_1h[i] or rsi14_1h[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for pullback entry in direction of 4h trend and 1d regime
            pullback_to_ema = abs(close[i] - ema21_1h[i]) / ema21_1h[i] < 0.005  # Within 0.5% of EMA21
            
            if uptrend_4h and bullish_regime and pullback_to_ema and rsi14_1h[i] < 30:
                position = 1
                signals[i] = 0.20
            elif downtrend_4h and bearish_regime and pullback_to_ema and rsi14_1h[i] > 70:
                position = -1
                signals[i] = -0.20
    
    return signals