#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly trend filter (1w EMA50) and volume confirmation.
# Long when price breaks above 20-period Donchian high AND weekly trend is up (close > EMA50) AND volume > 1.5x daily average volume.
# Short when price breaks below 20-period Donchian low AND weekly trend is down (close < EMA50) AND volume > 1.5x daily average volume.
# Exit when price crosses back through the Donchian midpoint.
# Uses Donchian for trend following structure, weekly EMA for trend filter to avoid counter-trend trades, volume for confirmation.
# Target: 15-30 trades/year per symbol.
name = "6h_Donchian_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_close = df_1w['close'].iloc[-1] if len(df_1w) > 0 else 0  # Not used directly, using aligned EMA
        ema50_val = ema50_1w_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        upper = high_roll[i]
        lower = low_roll[i]
        mid = donchian_mid[i]
        
        # Trend filter: weekly EMA50 slope (using current vs previous aligned value)
        if i > start_idx:
            ema50_prev = ema50_1w_aligned[i-1]
            weekly_trend_up = ema50_val > ema50_prev
            weekly_trend_down = ema50_val < ema50_prev
        else:
            weekly_trend_up = True
            weekly_trend_down = False
        
        if position == 0:
            # Long entry: break above upper band + weekly uptrend + volume spike
            if price > upper and weekly_trend_up and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + weekly downtrend + volume spike
            elif price < lower and weekly_trend_down and vol > 1.5 * vol_ma:
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