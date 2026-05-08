#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level + weekly EMA > 50-period EMA + volume spike
# Short when price breaks below Camarilla S3 level + weekly EMA < 50-period EMA + volume spike
# Exit when price crosses Camarilla H4/L4 level or weekly trend reverses
# Uses Camarilla pivot levels for institutional support/resistance, weekly trend filter for direction bias
# Targets 15-25 trades/year to minimize fee drag while capturing significant breakouts

name = "12h_Camarilla_R3S3_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    # H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    # We'll use R3/S3 for breakout and H4/L4 for exit
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros(len(df_daily))
    camarilla_s3 = np.zeros(len(df_daily))
    camarilla_h4 = np.zeros(len(df_daily))
    camarilla_l4 = np.zeros(len(df_daily))
    
    for i in range(len(df_daily)):
        high_low = daily_high[i] - daily_low[i]
        camarilla_r3[i] = daily_close[i] + 1.1 * high_low / 4
        camarilla_s3[i] = daily_close[i] - 1.1 * high_low / 4
        camarilla_h4[i] = daily_close[i] + 1.1 * high_low / 2
        camarilla_l4[i] = daily_close[i] - 1.1 * high_low / 2
    
    # Align weekly EMA to 12h timeframe
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_50)
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l4)
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_20 = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_ema_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        vol_filter = False
        if idx_daily >= 0:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and trend alignment
            # Long: price breaks above Camarilla R3 + weekly uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and weekly_ema_50_aligned[i] > weekly_ema_50_aligned[i-1]:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Camarilla S3 + weekly downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and weekly_ema_50_aligned[i] < weekly_ema_50_aligned[i-1]:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla L4 or weekly trend turns down
            if close[i] < camarilla_l4_aligned[i] or weekly_ema_50_aligned[i] < weekly_ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla H4 or weekly trend turns up
            if close[i] > camarilla_h4_aligned[i] or weekly_ema_50_aligned[i] > weekly_ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals