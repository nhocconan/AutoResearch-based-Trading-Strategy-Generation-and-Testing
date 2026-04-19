#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation
# Uses weekly Donchian channels to establish long-term trend direction
# Enters on daily breakouts in the direction of the weekly trend with volume confirmation
# Weekly trend filter reduces false breakouts in choppy markets
# Target: 10-25 trades/year to minimize fee drag while capturing major moves
name = "1d_DonchianBreakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels for trend determination (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    weekly_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Daily Donchian channels for entry signals (20-period)
    daily_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    daily_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for position sizing and volatility filter
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x average volume (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(daily_high[i]) or np.isnan(daily_low[i]) or np.isnan(atr_daily[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_daily[i]
        
        # Volume confirmation
        volume_filter = volume[i] > 1.3 * volume_ma[i]
        
        # Weekly trend determination
        # Uptrend: price above weekly midpoint, Downtrend: price below weekly midpoint
        weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
        weekly_uptrend = price > weekly_mid
        weekly_downtrend = price < weekly_mid
        
        if position == 0:
            # Long: Daily breakout above upper band + weekly uptrend + volume
            if price > daily_high[i] and weekly_uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Daily breakout below lower band + weekly downtrend + volume
            elif price < daily_low[i] and weekly_downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Daily breakdown below lower band OR weekly trend reversal
            if price < daily_low[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Daily breakout above upper band OR weekly trend reversal
            if price > daily_high[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals