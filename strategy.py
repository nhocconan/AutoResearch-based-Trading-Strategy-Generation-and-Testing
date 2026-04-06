#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(200) trend filter and volume confirmation.
# Donchian breakout captures momentum in trending markets.
# EMA(200) on 1d filters trades to align with long-term trend, reducing counter-trend entries.
# Volume > 1.5x 20-period average confirms momentum behind breakout.
# Works in bull via breakouts in uptrend, in bear via breakdowns in downtrend.
# Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag.

name = "4h_donchian20_1dema200_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price closes below Donchian lower band OR price below EMA200
            if close[i] <= lowest_low_20[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper band OR price above EMA200
            if close[i] >= highest_high_20[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA200 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_200_aligned[i]:
                    # Bullish breakout with 1d uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_200_aligned[i]:
                    # Bearish breakdown with 1d downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(200) trend filter and volume confirmation.
# Donchian breakout captures momentum in trending markets.
# EMA(200) on 1d filters trades to align with long-term trend, reducing counter-trend entries.
# Volume > 1.5x 20-period average confirms momentum behind breakout.
# Works in bull via breakouts in uptrend, in bear via breakdowns in downtrend.
# Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag.

name = "4h_donchian20_1dema200_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price closes below Donchian lower band OR price below EMA200
            if close[i] <= lowest_low_20[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper band OR price above EMA200
            if close[i] >= highest_high_20[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA200 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_200_aligned[i]:
                    # Bullish breakout with 1d uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_200_aligned[i]:
                    # Bearish breakdown with 1d downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals