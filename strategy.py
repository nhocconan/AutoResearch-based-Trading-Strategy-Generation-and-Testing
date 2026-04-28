#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 1d Donchian upper with 1w EMA50 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below 1d Donchian lower with 1w EMA50 downtrend and volume confirmation.
# Exit when price retraces to the 1d Donchian midpoint (upper+lower)/2.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 30-100 total trades over 4 years (7-25/year).
# Donchian channels provide robust price structure. EMA50 on 1w ensures primary trend alignment.
# Volume confirmation filters weak breakouts. Works in both bull (trend continuation) and bear (mean reversion via midpoint exit).

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (using current day's data)
    # For Donchian, we need the current bar's high/low to calculate the channel
    # But to avoid look-ahead, we use the previous completed bar's Donchian levels
    # So we calculate Donchian on lagged data
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1w EMA (50-period)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 1d (shifted by one bar to avoid look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d Donchian channels (20-period) using lagged data to avoid look-ahead
    # We use the previous 20 completed 1d bars to calculate the channel for the current bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian upper = max(high of previous 20 bars)
    # Donchian lower = min(low of previous 20 bars)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need at least 21 bars for Donchian (20 for calculation + 1 for shift)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_aligned[i]
        ema_trend_down = close[i] < ema_50_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian upper, price > EMA50 (uptrend), volume confirm
            if price > donchian_upper[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian lower, price < EMA50 (downtrend), volume confirm
            elif price < donchian_lower[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals