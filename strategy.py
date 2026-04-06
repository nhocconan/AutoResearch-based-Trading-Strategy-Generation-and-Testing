#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(10) trend + volume confirmation
# Long when price breaks above 20-day high AND price > 10-week EMA AND volume > 2x average
# Short when price breaks below 20-day low AND price < 10-week EMA AND volume > 2x average
# Exit when price crosses opposite Donchian band or volume drops below average
# Uses 1d timeframe for clear breakout signals with low trade frequency (target: 30-100 total trades over 4 years)
# Works in both bull/bear markets by following weekly trend with volume confirmation

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 1d
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 10-week EMA on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_10w = pd.Series(weekly_close).ewm(span=10, min_periods=10, adjust=False).mean()
    ema_10w_aligned = align_htf_to_ltf(prices, df_1w, ema_10w)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_10w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses opposite Donchian band OR volume drops below average
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            # Long: price breaks above 20-day high AND price > 10-week EMA AND volume > 2x average
            if (close[i] > highest_high[i] and close[i] > ema_10w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND price < 10-week EMA AND volume > 2x average
            elif (close[i] < lowest_low[i] and close[i] < ema_10w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(10) trend + volume confirmation
# Long when price breaks above 20-day high AND price > 10-week EMA AND volume > 2x average
# Short when price breaks below 20-day low AND price < 10-week EMA AND volume > 2x average
# Exit when price crosses opposite Donchian band or volume drops below average
# Uses 1d timeframe for clear breakout signals with low trade frequency (target: 30-100 total trades over 4 years)
# Works in both bull/bear markets by following weekly trend with volume confirmation

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 1d
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 10-week EMA on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_10w = pd.Series(weekly_close).ewm(span=10, min_periods=10, adjust=False).mean()
    ema_10w_aligned = align_htf_to_ltf(prices, df_1w, ema_10w)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_10w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses opposite Donchian band OR volume drops below average
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            # Long: price breaks above 20-day high AND price > 10-week EMA AND volume > 2x average
            if (close[i] > highest_high[i] and close[i] > ema_10w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND price < 10-week EMA AND volume > 2x average
            elif (close[i] < lowest_low[i] and close[i] < ema_10w_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals