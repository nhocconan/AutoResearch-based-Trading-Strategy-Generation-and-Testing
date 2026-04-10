#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian upper (20) AND 1d volume > 1.3x 20-bar avg AND 1w close > 1w open
# - Short when price breaks below Donchian lower (20) AND 1d volume > 1.3x 20-bar avg AND 1w close < 1w open
# - Exit when price returns to Donchian middle (20) or opposite Donchian band touched
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian channels provide structural breakouts; volume confirms conviction; weekly trend filter reduces counter-trend trades

name = "1d_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower: 20-period high/low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
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
    
    for i in range(20, n):  # Start after Donchian warmup
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
            # Long when price breaks above Donchian upper AND volume spike AND weekly bullish
            if (close[i] > donchian_upper[i] and 
                vol_spike[i] and 
                weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND volume spike AND weekly bearish
            elif (close[i] < donchian_lower[i] and 
                  vol_spike[i] and 
                  weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian middle OR touches opposite band (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if close[i] <= donchian_middle[i] or close[i] <= donchian_lower[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if close[i] >= donchian_middle[i] or close[i] >= donchian_upper[i]:
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