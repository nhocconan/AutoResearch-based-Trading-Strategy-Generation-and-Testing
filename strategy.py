#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band Squeeze with Daily Breakout and Volume Confirmation
# Bollinger Band Width < 20th percentile indicates low volatility squeeze
# Breakout above upper BB with volume > 1.5x 20-day average signals new trend
# Works in bull markets (breakouts continue) and bear markets (false breakdowns fade)
# Weekly timeframe for squeeze detection reduces false signals
# Daily timeframe for execution balances responsiveness and trade frequency
# Target: 15-25 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for Bollinger Bands
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(weekly_close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(weekly_close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    bb_width = upper_bb - lower_bb
    
    # Weekly Bollinger Band Width percentile (20-day lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB Width < 20th percentile
    squeeze = bb_width_percentile < 20
    
    # Align squeeze signal to daily
    squeeze_aligned = align_htf_to_ltf(prices, df_weekly, squeeze.astype(float))
    
    # Daily Bollinger Bands for breakout
    daily_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    daily_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    daily_upper = daily_sma + (2 * daily_std)
    daily_lower = daily_sma - (2 * daily_std)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(40, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(daily_upper[i]) or 
            np.isnan(daily_lower[i]) or np.isnan(volume_confirm[i])):
            continue
        
        # Long entry: squeeze + breakout above upper BB + volume
        if (squeeze_aligned[i] > 0.5 and  # squeezed state
            close[i] > daily_upper[i] and 
            volume_confirm[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: squeeze + breakdown below lower BB + volume
        elif (squeeze_aligned[i] > 0.5 and  # squeezed state
              close[i] < daily_lower[i] and 
              volume_confirm[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: price returns to middle of Bollinger Bands
        elif position == 1 and close[i] < daily_sma[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > daily_sma[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyBB_Squeeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0