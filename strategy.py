#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price crosses back through 20-day opposite level or weekly trend reverses
# Uses 1d timeframe with 1w trend filter to reduce false breaks, targeting 50-150 total trades over 4 years
# Works in both bull/bear markets by requiring trend alignment for breakouts

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
    
    # Donchian channels (20-period) on daily timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_low[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_high[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend alignment and volume
            # Long: price breaks above 20-day high AND weekly trend up AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_ema_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly trend down AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_ema_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA20) AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA20) AND volume > 1.5x average
# Exit when price