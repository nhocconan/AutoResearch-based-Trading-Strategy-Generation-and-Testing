# 12h_Donchian_Breakout_DailyTrend_Volume
# Hypothesis: 12-hour Donchian breakout with daily trend filter and volume confirmation.
# Long when price breaks above 20-period 12h high AND price > daily EMA50 AND volume > 1.5x daily average volume
# Short when price breaks below 20-period 12h low AND price < daily EMA50 AND volume > 1.5x daily average volume
# Exit when price crosses back below/above 20-period 12h moving average or when daily trend reverses
# Target: 12-37 trades/year per symbol to stay within frequency limits for 12h timeframe.
# Uses Donchian for breakout signals, daily EMA for trend filter, volume for confirmation.
# Works in both bull and bear markets due to directional bias from daily trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channels (20-period)
    high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get daily data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    daily_ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Get daily average volume for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        daily_ema = daily_ema50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above 20-period 12h high AND daily uptrend AND volume confirmation
            if price > upper_band and price > daily_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period 12h low AND daily downtrend AND volume confirmation
            elif price < lower_band and price < daily_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-period 12h moving average OR daily trend turns bearish
            mid_point = (upper_band + lower_band) / 2
            if price < mid_point or price < daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-period 12h moving average OR daily trend turns bullish
            mid_point = (upper_band + lower_band) / 2
            if price > mid_point or price > daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals