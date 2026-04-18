# 1d_Weekly_Donchian_Breakout_Volume_Trend
# Hypothesis: Weekly Donchian breakouts with volume confirmation and 1d trend filter capture major trends while minimizing trades.
# Uses weekly high/low breakouts (20-week lookback) to catch sustained moves, volume to confirm conviction,
# and 1d EMA to ensure alignment with intermediate trend. Low frequency (~10-20 trades/year) reduces fee drag.
# Works in bull markets (catching uptrends) and bear markets (catching downtrends) by only taking breakouts
# aligned with 1d EMA trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # 1d EMA for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_high_level = donchian_high_aligned[i]
        weekly_low_level = donchian_low_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Look for weekly Donchian breakout with volume, in trend direction
            if vol_ok:
                # Breakout above weekly high with volume in uptrend
                if price > weekly_high_level and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Breakout below weekly low with volume in downtrend
                elif price < weekly_low_level and price < trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if price returns to weekly low or trend reverses
            if price < weekly_low_level or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly high or trend reverses
            if price > weekly_high_level or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0