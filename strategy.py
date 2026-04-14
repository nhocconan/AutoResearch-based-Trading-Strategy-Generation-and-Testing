# 1d_1w_Donchian20_WeeklyTrend_Breakout
# Strategy: Buy when price breaks above 20-day high in bullish weekly trend, sell when breaks below 20-day low in bearish weekly trend.
# Weekly trend = price above/below 20-week EMA. Uses volume confirmation to filter breakouts.
# Designed for low trade frequency (<25/year) to avoid fee drag, works in bull/bear via trend filter.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        ema_series = pd.Series(close_1w)
        ema20_1w = ema_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 20-day Donchian channels (high/low)
    highest_20d = np.full(n, np.nan)
    lowest_20d = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        highest_20d = high_series.rolling(window=20, min_periods=20).max().values
        lowest_20d = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for confirmation
    avg_vol_20d = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        avg_vol_20d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if np.isnan(highest_20d[i]) or np.isnan(lowest_20d[i]) or np.isnan(avg_vol_20d[i]) or np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume > 20-day average
        if volume[i] <= avg_vol_20d[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above 20-day high in bullish weekly trend (price > weekly EMA20)
            if high[i] > highest_20d[i] and close[i] > ema20_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: Break below 20-day low in bearish weekly trend (price < weekly EMA20)
            elif low[i] < lowest_20d[i] and close[i] < ema20_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below 20-day low OR trend turns bearish
            if low[i] < lowest_20d[i] or close[i] < ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above 20-day high OR trend turns bullish
            if high[i] > highest_20d[i] or close[i] > ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_WeeklyTrend_Breakout"
timeframe = "1d"
leverage = 1.0