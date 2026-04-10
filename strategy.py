#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian upper band (20-day high) AND 1d volume > 1.3x 20-bar avg AND 1w close > 1w open (bullish weekly candle)
# - Short when price breaks below Donchian lower band (20-day low) AND 1d volume > 1.3x 20-bar avg AND 1w close < 1w open (bearish weekly candle)
# - Exit when price returns to Donchian middle band (20-day average of high and low)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian channels provide clear trend structure; volume confirms breakout strength
# - Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws

name = "1d_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20): upper = 20-period high, lower = 20-period low, middle = average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.3 * volume_20_avg)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(vol_spike[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper band AND volume spike AND weekly bullish
            if (prices['close'].iloc[i] > donchian_upper[i] and 
                vol_spike[i] and 
                weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower band AND volume spike AND weekly bearish
            elif (prices['close'].iloc[i] < donchian_lower[i] and 
                  vol_spike[i] and 
                  weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian middle band (mean reversion to equilibrium)
            # Exit when price returns to Donchian middle band
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= donchian_middle[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= donchian_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals