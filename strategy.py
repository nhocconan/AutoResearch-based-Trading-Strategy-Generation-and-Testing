#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above daily 20-period Donchian high AND weekly price > weekly SMA(50) AND daily volume > 1.5x daily average volume
# Short when price breaks below daily 20-period Donchian low AND weekly price < weekly SMA(50) AND daily volume > 1.5x daily average volume
# Exit when price crosses back through the Donchian midpoint
# Uses daily price structure, weekly trend filter for regime, volume for confirmation.
# Target: 10-25 trades/year per symbol (30-100 total over 4 years).
name = "1d_Donchian_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter (SMA 50)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Get daily average volume for confirmation
    vol_ma_daily = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_sma50_aligned[i]) or 
            np.isnan(vol_ma_daily[i]) or 
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or 
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_trend = weekly_sma50_aligned[i]
        vol_ma = vol_ma_daily[i]
        vol = volume[i]
        upper = high_roll[i]
        lower = low_roll[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long entry: break above upper band + weekly uptrend + volume spike
            if price > upper and price > weekly_trend and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + weekly downtrend + volume spike
            elif price < lower and price < weekly_trend and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals