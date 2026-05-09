#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout (20-week high/low) with daily volume confirmation and trend filter.
# Weekly Donchian provides strong structural breaks; volume confirms institutional interest; daily trend filter avoids counter-trend trades.
# Designed for low trade frequency (<25/year) to minimize fee drag, works in bull/bear via trend filter.
name = "1d_WeeklyDonchian20_Breakout_DailyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient history for weekly lookback
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (20-period high/low)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian high and low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily trend filter: 50-period EMA on daily close
    ema_50_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_daily[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        d_high = donchian_high_aligned[i]
        d_low = donchian_low_aligned[i]
        ema_50 = ema_50_daily[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > weekly Donchian high AND price > daily EMA50 (uptrend) AND volume > 1.5x average
            if close[i] > d_high and close[i] > ema_50 and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < weekly Donchian low AND price < daily EMA50 (downtrend) AND volume > 1.5x average
            elif close[i] < d_low and close[i] < ema_50 and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < weekly Donchian low OR trend reverses (price < daily EMA50)
            if close[i] < d_low or close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > weekly Donchian high OR trend reverses (price > daily EMA50)
            if close[i] > d_high or close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals