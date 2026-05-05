#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout direction + 1d trend filter + volume confirmation
# Long when: price breaks above 4h Donchian upper (20) AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when: price breaks below 4h Donchian lower (20) AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when: price crosses back through the 4h Donchian midpoint (mean reversion logic)
# Uses 4h structure for direction (lower frequency = fewer trades), 1h for precise entry timing, 1d for trend filter.
# Designed to work in both bull (continuation breakouts) and bear (failed breakdowns = short opportunities) markets.
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years = 15-37/year.

name = "1h_DonchianBreakout_4hDir_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values  # datetime64[ms] for session filter
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    if len(high_4h) >= 20:
        # Donchian upper: highest high over last 20 periods
        donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
        # Donchian lower: lowest low over last 20 periods
        donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
        # Donchian midpoint: average of upper and lower
        donchian_mid = (donchian_high_20 + donchian_low_20) / 2.0
    else:
        donchian_high_20 = np.full(len(high_4h), np.nan)
        donchian_low_20 = np.full(len(high_4h), np.nan)
        donchian_mid = np.full(len(high_4h), np.nan)
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h Donchian upper AND above 1d EMA50 AND volume confirmation
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian lower AND below 1d EMA50 AND volume confirmation
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals