#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA(50) AND volume > 1.5x average
# Short when price breaks below 20-day low AND price < 1w EMA(50) AND volume > 1.5x average
# Exit when price crosses 10-day EMA in opposite direction or volatility expands
# Uses daily timeframe to target 30-100 trades over 4 years, weekly trend filter to avoid counter-trend trades
# Works in bull markets via breakouts and in bear via short breakdowns with trend alignment

name = "1d_donchian20_1w_ema_vol_v10"
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
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Daily EMA(10) for exit
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Weekly EMA(50) for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_50_w = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_10[i]) or \
           np.isnan(ema_50_w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < ema_10[i]:  # price crosses below EMA(10)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > ema_10[i]:  # price crosses above EMA(10)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend alignment and volume confirmation
            # Long: price breaks above 20-day high AND above weekly EMA(50) + volume
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_w_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND below weekly EMA(50) + volume
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_w_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA(50) AND volume > 1.5x average
# Short when price breaks below 20-day low AND price < 1w EMA(50) AND volume > 1.5x average
# Exit when price crosses 10-day EMA in opposite direction or volatility expands
# Uses daily timeframe to target 30-100 trades over 4 years, weekly trend filter to avoid counter-trend trades
# Works in bull markets via breakouts and in bear via short breakdowns with trend alignment

name = "1d_donchian20_1w_ema_vol_v10"
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
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Daily EMA(10) for exit
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Weekly EMA(50) for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_50_w = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_10[i]) or \
           np.isnan(ema_50_w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < ema_10[i]:  # price crosses below EMA(10)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > ema_10[i]:  # price crosses above EMA(10)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend alignment and volume confirmation
            # Long: price breaks above 20-day high AND above weekly EMA(50) + volume
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_w_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND below weekly EMA(50) + volume
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_w_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals