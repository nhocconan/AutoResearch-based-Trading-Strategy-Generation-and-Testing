#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d EMA trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back through Donchian midpoint or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear markets by combining trend filter (EMA) with breakout logic

name = "12h_donchian_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Donchian Channel (20-period) - price channel breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint OR EMA condition fails
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA trend filter + volume confirmation
            # Long: price breaks above Donchian high AND above EMA(50) + volume confirmation
            if (close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below EMA(50) + volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extremes + EMA trend filter + volume confirmation
            # Long: Williams %R oversold (< -80) AND price above EMA(50) + volume confirmation
            if (williams_r[i] < -80 and close[i] > ema_50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA(50) + volume confirmation
            elif (williams_r[i] > -20 and close[i] < ema_50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when Williams %R > -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or EMA condition fails
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Williams %R is effective in ranging markets; EMA filter adds trend bias for better performance

name = "12h_williamsr_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 OR EMA condition fails
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] <= ema_5