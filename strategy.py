#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend filter
        uptrend = close_1w_s.iloc[-1] > ema_50_1w[-1] if len(close_1w_s) > 0 else False  # placeholder - will fix below
        
        # Fix: properly get current 1w values
        # Since we're using aligned arrays, we need to get the 1w values for current bar
        # We'll recalculate the trend inside loop using the aligned EMA
        
    # Rewriting loop with correct logic
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend filter using aligned EMA
        # Need current 1w close price - we'll approximate using the fact that
        # aligned arrays give us the completed 1w bar's EMA value
        # For trend, we compare 1w close to its EMA50
        # Since we don't have 1w close aligned, we'll use price action:
        # In uptrend: price tends to be above EMA
        # We'll use a heuristic: if current 12h price > aligned 1w EMA50, consider bullish bias
        # But better: calculate 1w close separately
        
        # Let's load 1w close and align it
        # Actually, let's restructure - load close_1w and align it
        
    # Complete rewrite
    signals = np.zeros(n)
    position = 0
    
    # Pre-calculate all needed 1w data aligned
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    # For EMA comparison, we need the 1w EMA values aligned
    # We already have ema_50_1w_aligned
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        # Using aligned arrays gives us completed 1w bar values
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bearish
            if williams_r[i] > -50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bullish
            if williams_r[i] < -50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if bullish_trend and volume_confirmed:
                # Enter long when Williams %R is oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif bearish_trend and volume_confirmed:
                # Enter short when Williams %R is overbought (> -20) in downtrend
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals

# Fix the EMA calculation - it was using the wrong variable
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_period=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_period=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bearish
            if williams_r[i] > -50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bullish
            if williams_r[i] < -50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if bullish_trend and volume_confirmed:
                # Enter long when Williams %R is oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif bearish_trend and volume_confirmed:
                # Enter short when Williams %R is overbought (> -20) in downtrend
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_period=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_period=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bearish
            if williams_r[i] > -50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bullish
            if williams_r[i] < -50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if bullish_trend and volume_confirmed:
                # Enter long when Williams %R is oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif bearish_trend and volume_confirmed:
                # Enter short when Williams %R is overbought (> -20) in downtrend
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_period=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_period=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bearish
            if williams_r[i] > -50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bullish
            if williams_r[i] < -50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if bullish_trend and volume_confirmed:
                # Enter long when Williams %R is oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif bearish_trend and volume_confirmed:
                # Enter short when Williams %R is overbought (> -20) in downtrend
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_period=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_period=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend changes to bearish
            if williams_r[i] > -50 or bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend changes to bullish
            if williams_r[i] < -50 or bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if bullish_trend and volume_confirmed:
                # Enter long when Williams %R is oversold (< -80) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif bearish_trend and volume_confirmed:
                # Enter short when Williams %R is overbought (> -20) in downtrend
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# In 1w uptrend (price > EMA50): look for Williams %R oversold (< -80) to go long
# In 1w downtrend (price < EMA50): look for Williams %R overbought (> -20) to go short
# Uses volume confirmation (> 1.5x 20-period average) to filter false signals
# Discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1w_williamsr_meanreversion_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_period=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_period=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        # 1w trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        bearish_trend = close_1w_aligned[i] < ema