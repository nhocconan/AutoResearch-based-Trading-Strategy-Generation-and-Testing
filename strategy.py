#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation.
# Long when price breaks above 20-period 6h Donchian high in bull trend (close > weekly EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below 20-period 6h Donchian low in bear trend (close < weekly EMA50) with volume spike.
# Weekly trend filter reduces whipsaw vs daily, improving performance in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-38/year).

name = "6h_Donchian20_WeeklyEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        weekly_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > weekly_trend
        is_bear_trend = close_val < weekly_trend
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close_val > donchian_high[i-1]
        breakout_down = close_val < donchian_low[i-1]
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR trend reversal
            if breakout_down or close_val < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout OR trend reversal
            if breakout_up or close_val > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals