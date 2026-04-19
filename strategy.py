#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly 52-Week High Breakout with Volume Confirmation and Weekly Trend Filter
# Targets long-term breakouts in both bull and bear markets using weekly structure.
# In bull markets: buys breakouts above weekly 52-week high with volume confirmation.
# In bear markets: shorts breakdowns below weekly 52-week low with volume confirmation.
# Uses weekly EMA200 as trend filter to avoid counter-trend trades.
# Weekly timeframe reduces trade frequency to avoid fee drag (target: 10-30 trades/year).
name = "1d_Weekly52WeekHighBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for multi-timeframe analysis (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly close for 52-week high/low and EMA200
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # 52-week high and low (approximately 252 trading days / 5 = ~50 weeks)
    # Using 50-week lookback for 52-week high/low
    lookback_weeks = 50
    high_52w = pd.Series(high_weekly).rolling(window=lookback_weeks, min_periods=lookback_weeks).max().values
    low_52w = pd.Series(low_weekly).rolling(window=lookback_weeks, min_periods=lookback_weeks).min().values
    
    # Weekly EMA200 for trend filter
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    high_52w_aligned = align_htf_to_ltf(prices, df_weekly, high_52w)
    low_52w_aligned = align_htf_to_ltf(prices, df_weekly, low_52w)
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Daily ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for weekly indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_52w_aligned[i]) or np.isnan(low_52w_aligned[i]) or \
           np.isnan(ema200_weekly_aligned[i]) or np.isnan(atr_daily[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_daily[i]
        
        # Volume filter: current volume > 2.0x average volume (50-day)
        if i >= 50:
            avg_volume = np.mean(volume[i-50:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 2.0 * avg_volume
        
        if position == 0:
            # Long: breakout above 52-week high + volume + weekly uptrend
            if high[i] > high_52w_aligned[i-1] and volume_filter and price > ema200_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 52-week low + volume + weekly downtrend
            elif low[i] < low_52w_aligned[i-1] and volume_filter and price < ema200_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below 52-week low or ATR-based stop
            if close[i] < low_52w_aligned[i] or close[i] < close[i-1] - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above 52-week high or ATR-based stop
            if close[i] > high_52w_aligned[i] or close[i] > close[i-1] + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals