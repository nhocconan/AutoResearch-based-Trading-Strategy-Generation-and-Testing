#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA25) and volume confirmation
# Long when price breaks above Donchian upper band AND 12h EMA25 rising AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND 12h EMA25 falling AND volume > 1.5x average
# Exit when price crosses opposite Donchian band or volume drops below average
# Uses 4h timeframe targeting 80-180 total trades over 4 years (20-45/year)
# Works in bull markets via trend-following breakouts and in bear markets via short breakdowns
# Volume confirmation reduces false breakouts, 12h EMA filter avoids counter-trend trades

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
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
    
    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # 12h EMA25 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):
        # Skip if required data not available
        if np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_up[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian upper AND 12h EMA25 rising AND volume confirmation
            if (close[i] > donchian_up[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h EMA25 falling AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA25) and volume confirmation
# Long when price breaks above Donchian upper band AND 12h EMA25 rising AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND 12h EMA25 falling AND volume > 1.5x average
# Exit when price crosses opposite Donchian band or volume drops below average
# Uses 4h timeframe targeting 80-180 total trades over 4 years (20-45/year)
# Works in bull markets via trend-following breakouts and in bear markets via short breakdowns
# Volume confirmation reduces false breakouts, 12h EMA filter avoids counter-trend trades

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
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
    
    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # 12h EMA25 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(25, n):
        # Skip if required data not available
        if np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_up[i] or volume[i] < volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian upper AND 12h EMA25 rising AND volume confirmation
            if (close[i] > donchian_up[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND 12h EMA25 falling AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals