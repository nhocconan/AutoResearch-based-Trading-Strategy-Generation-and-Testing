#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_1d_ema_rsi_meanrev_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calc session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate daily RSI(14) and EMA(50)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI(14) for entry timing
    delta_1h = np.diff(close, prepend=close[0])
    gain_1h = np.where(delta_1h > 0, delta_1h, 0)
    loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
    avg_gain_1h = pd.Series(gain_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1h = pd.Series(loss_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1h = avg_gain_1h / (avg_loss_1h + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs_1h))
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price_close = close[i]
        rsi_daily = rsi_1d_aligned[i]
        ema50_daily = ema50_1d_aligned[i]
        rsi_hourly = rsi_1h[i]
        
        # Mean reversion signals
        # Long when: daily RSI < 30 (oversold) + price below daily EMA50 + hourly RSI < 30
        # Short when: daily RSI > 70 (overbought) + price above daily EMA50 + hourly RSI > 70
        enter_long = False
        enter_short = False
        
        if in_session:
            if rsi_daily < 30 and price_close < ema50_daily and rsi_hourly < 30:
                enter_long = True
            if rsi_daily > 70 and price_close > ema50_daily and rsi_hourly > 70:
                enter_short = True
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = rsi_daily >= 40
        exit_short = rsi_daily <= 60
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Mean reversion on daily timeframe with hourly execution timing.
# Uses daily RSI(14) for regime (oversold/overbought) and price relative to daily EMA50 for trend filter.
# Hourly RSI provides entry timing precision. Session filter (08-20 UTC) reduces noise.
# Works in both bull and bear markets by fading extremes. Target: 60-150 trades over 4 years.
# Position size 0.20 limits drawdown. Discrete sizing minimizes fee churn.