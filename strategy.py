#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with bullish daily trend and volume > 20-period average.
# Short when price breaks below 20-period Donchian low with bearish daily trend and volume > 20-period average.
# Uses daily trend filter to avoid counter-trend trades. Volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need at least 20 for Donchian + buffer
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below Donchian low or daily turn bearish
            if (close[i] < donchian_low[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Donchian high or daily turn bullish
            if (close[i] > donchian_high[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with daily trend filter and volume confirmation
            # Long: price breaks above Donchian high during bullish day with volume confirmation
            if (close[i] > donchian_high[i] and 
                daily_bullish_aligned[i] and 
                volume[i] > volume_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low during bearish day with volume confirmation
            elif (close[i] < donchian_low[i] and 
                  daily_bearish_aligned[i] and 
                  volume[i] > volume_avg[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with bullish daily trend and volume > 20-period average.
# Short when price breaks below 20-period Donchian low with bearish daily trend and volume > 20-period average.
# Uses daily trend filter to avoid counter-trend trades. Volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need at least 20 for Donchian + buffer
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: bullish/bearish day based on close vs open
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_bullish = daily_close > daily_open  # True for bullish day
    daily_bearish = daily_close < daily_open   # True for bearish day
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily trend data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below Donchian low or daily turn bearish
            if (close[i] < donchian_low[i] or 
                daily_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above Donchian high or daily turn bullish
            if (close[i] > donchian_high[i] or 
                daily_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with daily trend filter and volume confirmation
            # Long: price breaks above Donchian high during bullish day with volume confirmation
            if (close[i] > donchian_high[i] and 
                daily_bullish_aligned[i] and 
                volume[i] > volume_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low during bearish day with volume confirmation
            elif (close[i] < donchian_low[i] and 
                  daily_bearish_aligned[i] and 
                  volume[i] > volume_avg[i]):
                signals[i] = -0.25
                position = -1
    
    return signals