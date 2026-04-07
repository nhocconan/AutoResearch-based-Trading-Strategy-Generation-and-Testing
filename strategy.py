#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume Confirmation + 1d EMA Trend Filter
# Hypothesis: Buy breakouts above 4h Donchian(20) high in uptrend (price > daily EMA50),
# sell breakdowns below 4h Donchian(20) low in downtrend (price < daily EMA50),
# with volume confirmation (> 20-period average). Works in bull/bear by trading
# with daily trend. Target: 50-150 total trades over 4 years (12-37/year).

name = "4h_donchian_breakout_vol_1d_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    high_roll = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    low_roll = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Daily EMA(50) for trend filter
    close_daily = df_daily['close'].values
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donch_len, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if close[i] < low_roll[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if close[i] > high_roll[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume
            if vol_ok:
                if close[i] > ema_50_4h[i]:  # Uptrend
                    if high[i] > high_roll[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < low_roll[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals