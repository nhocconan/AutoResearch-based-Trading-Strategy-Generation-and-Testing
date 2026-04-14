#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with 1d RSI filter and volume confirmation.
# Weekly Donchian(20) breakout captures major trend direction, avoiding whipsaws in ranging markets.
# 1d RSI between 40-60 filters for neutral momentum to avoid overextended entries.
# Volume confirmation (>1.3x 50-period average) reduces false breakouts.
# Designed for both bull and bear markets by using weekly trend filter (price above/below weekly SMA50).
# Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donch_period = 20
    upper_donch = pd.Series(high_1w).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low_1w).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Calculate weekly SMA50 for trend filter
    sma_period = 50
    sma_1w = pd.Series(close_1w).rolling(window=sma_period, min_periods=sma_period).mean().values
    
    # Load daily data ONCE for RSI and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI (14-period)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to daily timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_1w, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1w, lower_donch)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.3x average volume (50-period)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_period, sma_period, rsi_period, 50)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donch_aligned[i]) or 
            np.isnan(lower_donch_aligned[i]) or
            np.isnan(sma_1w_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above/below weekly SMA50
        above_weekly_sma = close[i] > sma_1w_aligned[i]
        below_weekly_sma = close[i] < sma_1w_aligned[i]
        
        # RSI filter: avoid overextended conditions (40-60 range)
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band + uptrend + RSI neutral + volume
            if (close[i] > upper_donch_aligned[i] and 
                above_weekly_sma and 
                rsi_neutral and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly Donchian lower band + downtrend + RSI neutral + volume
            elif (close[i] < lower_donch_aligned[i] and 
                  below_weekly_sma and 
                  rsi_neutral and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly Donchian middle or trend reverses
            mid_donch = (upper_donch_aligned[i] + lower_donch_aligned[i]) / 2
            if (close[i] <= mid_donch or 
                below_weekly_sma):  # Trend reversal to downside
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly Donchian middle or trend reverses
            mid_donch = (upper_donch_aligned[i] + lower_donch_aligned[i]) / 2
            if (close[i] >= mid_donch or 
                above_weekly_sma):  # Trend reversal to upside
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wDonchian_RSI_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0