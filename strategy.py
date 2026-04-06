# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Long when price breaks above 6h Donchian upper band AND 1d EMA50 is rising AND volume > 1.5x 20-period average
# Short when price breaks below 6h Donchian lower band AND 1d EMA50 is falling AND volume > 1.5x 20-period average
# Exit when price touches opposite Donchian band or trend reverses
# Target: 50-150 trades over 4 years by combining trend, momentum, and volume filters
# Uses 1d EMA50 to avoid counter-trend trades and focus on higher timeframe momentum

name = "6h_donchian20_1d_ema50_vol_v1"
timeframe = "6h"
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
    
    # Donchian(20) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # EMA50 on 1d (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]  # handle first element
    ema_50_rising = ema_50_1d > ema_50_1d_prev
    ema_50_falling = ema_50_1d < ema_50_1d_prev
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches lower Donchian band OR trend turns bearish
            if close[i] <= donchian_low[i] or ema_50_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian band OR trend turns bullish
            if close[i] >= donchian_high[i] or ema_50_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and ema_50_rising_aligned[i]:
                    # Bullish breakout with bullish trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and ema_50_falling_aligned[i]:
                    # Bearish breakout with bearish trend
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Long when price breaks above 6h Donchian upper band AND 1d EMA50 is rising AND volume > 1.5x 20-period average
# Short when price breaks below 6h Donchian lower band AND 1d EMA50 is falling AND volume > 1.5x 20-period average
# Exit when price touches opposite Donchian band or trend reverses
# Target: 50-150 trades over 4 years by combining trend, momentum, and volume filters
# Uses 1d EMA50 to avoid counter-trend trades and focus on higher timeframe momentum

name = "6h_donchian20_1d_ema50_vol_v1"
timeframe = "6h"
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
    
    # Donchian(20) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # EMA50 on 1d (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]  # handle first element
    ema_50_rising = ema_50_1d > ema_50_1d_prev
    ema_50_falling = ema_50_1d < ema_50_1d_prev
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches lower Donchian band OR trend turns bearish
            if close[i] <= donchian_low[i] or ema_50_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian band OR trend turns bullish
            if close[i] >= donchian_high[i] or ema_50_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and ema_50_rising_aligned[i]:
                    # Bullish breakout with bullish trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and ema_50_falling_aligned[i]:
                    # Bearish breakout with bearish trend
                    signals[i] = -0.25
                    position = -1
    
    return signals