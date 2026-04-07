#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly mean reversion with daily trend filter and volume confirmation
# Uses weekly RSI(14) extremes (>80 oversold, <20 overbought) for mean reversion signals,
# filtered by daily trend (close > daily EMA(50) for longs, close < daily EMA(50) for shorts)
# and volume > 1.5x 20-period average for confirmation.
# Designed for low trade frequency (~15-30 trades/year) to minimize fee drag.
# Works in bull markets via trend-following longs and in bear markets via mean-reversion shorts.

name = "weekly_mean_reversion_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for mean reversion signal
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Mean reversion conditions from weekly RSI
        oversold = rsi_1w_aligned[i] < 20  # Extreme oversold
        overbought = rsi_1w_aligned[i] > 80  # Extreme overbought
        
        # Trend filter from daily EMA(50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long: weekly oversold + daily uptrend + volume confirmation
        if oversold and uptrend and vol_confirmed:
            signals[i] = 0.25
        # Short: weekly overbought + daily downtrend + volume confirmation
        elif overbought and downtrend and vol_confirmed:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals