#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(15) breakout with 1d trend filter (1d EMA21) and volume confirmation (1.5x avg volume)
# Long when price breaks above Donchian high + 1d EMA21 up + volume > 1.5x average
# Short when price breaks below Donchian low + 1d EMA21 down + volume > 1.5x average
# Exit when price crosses Donchian midpoint or 1d EMA21 reverses
# Target: 75-200 total trades over 4 years (19-50/year) by using moderate breakout conditions with trend filter
# Works in trending markets by following breakouts with trend filter, avoids chop via EMA21 filter

name = "4h_donchian_1d_ema_vol_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (15-period for more signals)
    donch_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donch_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint or 1d EMA21
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + 1d EMA21 up + volume
            if (close[i] > donch_high[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + 1d EMA21 down + volume
            elif (close[i] < donch_low[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(15) breakout with 1d trend filter (1d EMA21) and volume confirmation (1.5x avg volume)
# Long when price breaks above Donchian high + 1d EMA21 up + volume > 1.5x average
# Short when price breaks below Donchian low + 1d EMA21 down + volume > 1.5x average
# Exit when price crosses Donchian midpoint or 1d EMA21 reverses
# Target: 75-200 total trades over 4 years (19-50/year) by using moderate breakout conditions with trend filter
# Works in trending markets by following breakouts with trend filter, avoids chop via EMA21 filter

name = "4h_donchian_1d_ema_vol_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (15-period for more signals)
    donch_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donch_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d EMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint or 1d EMA21
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + 1d EMA21 up + volume
            if (close[i] > donch_high[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + 1d EMA21 down + volume
            elif (close[i] < donch_low[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals