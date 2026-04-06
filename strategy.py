#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter + volume confirmation + ATR stoploss
# Long when price breaks above Donchian upper (20-day high) AND price > 1w EMA AND volume > 1.5x average
# Short when price breaks below Donchian lower (20-day low) AND price < 1w EMA AND volume > 1.5x average
# Exit when price crosses 1w EMA in opposite direction OR price returns to Donchian middle
# Uses daily timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # need 20 for Donchian + 20 for EMA + buffer
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) - daily high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((donchian_upper + donchian_lower) / 2)
    
    # 1-week EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    ema_1w_values = ema_1w.values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # start after 40 to ensure all indicators ready
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses 1w EMA OR returns to Donchian middle
        if position == 1:  # long position
            if close[i] < ema_1w_aligned[i] or close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > ema_1w_aligned[i] or close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume
            # Long: break above upper band AND price > 1w EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_1w_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND price < 1w EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_1w_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter + volume confirmation + ATR stoploss
# Long when price breaks above Donchian upper (20-day high) AND price > 1w EMA AND volume > 1.5x average
# Short when price breaks below Donchian lower (20-day low) AND price < 1w EMA AND volume > 1.5x average
# Exit when price crosses 1w EMA in opposite direction OR price returns to Donchian middle
# Uses daily timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # need 20 for Donchian + 20 for EMA + buffer
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) - daily high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((donchian_upper + donchian_lower) / 2)
    
    # 1-week EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    ema_1w_values = ema_1w.values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # start after 40 to ensure all indicators ready
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses 1w EMA OR returns to Donchian middle
        if position == 1:  # long position
            if close[i] < ema_1w_aligned[i] or close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > ema_1w_aligned[i] or close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume
            # Long: break above upper band AND price > 1w EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_1w_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND price < 1w EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_1w_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>