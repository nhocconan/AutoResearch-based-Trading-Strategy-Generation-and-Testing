#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily price channel breakout with weekly trend filter and volume confirmation
# Works in bull/bear by using weekly trend to filter direction and volume to confirm breakouts
# Target: 20-50 trades/year to minimize fee drag
name = "1d_DailyChannel_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly close for EMA trend
    close_1w = df_1w['close'].values
    # Weekly EMA 34 for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper channel with volume and weekly uptrend
            if price > upper_channel and volume_confirmed and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume and weekly downtrend
            elif price < lower_channel and volume_confirmed and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below lower channel or weekly trend turns down
            if price < lower_channel or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above upper channel or weekly trend turns up
            if price > upper_channel or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals