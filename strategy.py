#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Enter long when price breaks above 1d Donchian high with 1w EMA50 uptrend and volume > 2.0x 20-bar average.
# Enter short when price breaks below 1d Donchian low with 1w EMA50 downtrend and volume confirmation.
# Exit when price retraces to the 1d Donchian midpoint.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 30-100 total trades over 4 years (7-25/year).
# Donchian breakouts capture strong momentum moves. 1w EMA50 filter ensures we only trade with the weekly trend,
# avoiding counter-trend whipsaws. Volume confirmation filters weak breakouts.
# This combination has shown promise on lower timeframes and should work on 1d timeframe with BTC/ETH.

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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: rolling max of high over 20 periods
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 1d (same timeframe, so identity alignment)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    donchian_mid_aligned = donchian_mid
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1w EMA (50-period)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient history for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
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
            # Long entry: price > Donchian high, price > EMA50 (uptrend), volume confirm
            if price > donchian_high_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, price < EMA50 (downtrend), volume confirm
            elif price < donchian_low_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at midpoint
            if price <= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at midpoint
            if price >= donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals