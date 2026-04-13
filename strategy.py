#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R mean reversion with 1w trend filter
    # Long: Williams %R < -80 (oversold) AND 1w close > 1w EMA200 (bullish trend)
    # Short: Williams %R > -20 (overbought) AND 1w close < 1w EMA200 (bearish trend)
    # Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
    # Using 1d for low trade frequency, Williams %R for mean reversion in ranging markets,
    # 1w EMA200 for trend filter to avoid counter-trend trades.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily Williams %R(14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        (highest_high - close) / (highest_high - lowest_low) * -100,
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: bullish if weekly close > weekly EMA200
        bullish_trend = close_1w[-1] > ema_200_1w[-1] if len(close_1w) == len(ema_200_1w) else False
        # Actually use aligned values for current bar
        # Need to get the weekly close value aligned to daily
        # Since we don't have weekly close aligned, we'll use the EMA alignment as proxy
        # Simpler approach: use the last available weekly close for trend
        if len(df_1w) > 0:
            weekly_close_now = df_1w['close'].iloc[-1]  # most recent weekly close
            weekly_ema_now = ema_200_1w[-1] if len(ema_200_1w) > 0 else 0
            bullish_trend = weekly_close_now > weekly_ema_now
            bearish_trend = weekly_close_now < weekly_ema_now
        else:
            bullish_trend = False
            bearish_trend = False
        
        # Entry logic: Williams %R extremes + trend filter
        long_entry = (williams_r[i] < -80) and bullish_trend
        short_entry = (williams_r[i] > -20) and bearish_trend
        
        # Exit logic: Williams %R crosses midpoint
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williamsr_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0