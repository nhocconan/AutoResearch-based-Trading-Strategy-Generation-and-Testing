#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Enter long when price breaks above 4h Donchian upper channel with 1d EMA200 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below 4h Donchian lower channel with 1d EMA200 downtrend and volume confirmation.
# Exit when price retraces to the 4h Donchian midpoint (upper+lower)/2.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian channels provide robust trend-following structure. EMA200 on 1d ensures alignment with higher timeframe trend.
# Volume confirmation filters weak breakouts. This pattern has shown strong performance on SOLUSDT and should work on BTC/ETH.

name = "4h_Donchian20_Breakout_1dEMA200_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:  # Need at least 20 4h bars for Donchian
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    # Middle channel = (upper + lower) / 2
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 4h (shifted by one bar to avoid look-ahead)
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (200-period)
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure sufficient history for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA200 trend filter: price > EMA200 = uptrend, price < EMA200 = downtrend
        ema_trend_up = close[i] > ema_200_aligned[i]
        ema_trend_down = close[i] < ema_200_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > EMA200 (uptrend), volume confirm
            if price > upper_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, price < EMA200 (downtrend), volume confirm
            elif price < lower_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals