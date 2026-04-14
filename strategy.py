#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with daily volume confirmation and weekly trend filter
# Long when price breaks above Camarilla H3 level with volume surge and weekly bullish trend
# Short when price breaks below Camarilla L3 level with volume surge and weekly bearish trend
# Exit when price crosses Camarilla H4/L4 levels or reverses at opposite pivot
# Uses weekly trend filter to avoid counter-trend trades and volume spike to confirm breakout strength
# Target: 20-50 trades per symbol over 4 years (5-12.5/year) to minimize fee drag
# Camarilla levels derived from prior day OHLC provide institutional reversal levels that work in both bull/bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla levels (based on previous day OHLC)
    if len(df_daily) < 1:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = daily_close + 1.1 * (daily_high - daily_low) / 2
    camarilla_h3 = daily_close + 1.1 * (daily_high - daily_low) / 4
    camarilla_l3 = daily_close - 1.1 * (daily_high - daily_low) / 4
    camarilla_l4 = daily_close - 1.1 * (daily_high - daily_low) / 2
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_l4)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations and daily alignment
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: break above Camarilla H3 with volume surge and weekly bullish trend
            if (price > camarilla_h3_aligned[i] and 
                vol_4h_current > 2.0 * vol_ma_daily_aligned[i] and  # Volume surge
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L3 with volume surge and weekly bearish trend
            elif (price < camarilla_l3_aligned[i] and 
                  vol_4h_current > 2.0 * vol_ma_daily_aligned[i] and  # Volume surge
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Camarilla L3 or reaches H4
            if price < camarilla_l3_aligned[i] or price > camarilla_h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Camarilla H3 or reaches L4
            if price > camarilla_h3_aligned[i] or price < camarilla_l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0