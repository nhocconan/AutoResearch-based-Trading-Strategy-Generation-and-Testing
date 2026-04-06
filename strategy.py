#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (20 for volume)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Fade at R3/S3 (counter-trend)
        # Long fade: price touches S3 and bounces up with volume
        fade_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and vol_filter[i]
        # Short fade: price touches R3 and rejects down with volume
        fade_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and vol_filter[i]
        
        # Breakout at R4/S4 (trend continuation)
        # Long breakout: price breaks above R4 with volume
        breakout_long = (close[i] > r4_1d_aligned[i]) and vol_filter[i]
        # Short breakdown: price breaks below S4 with volume
        breakout_short = (close[i] < s4_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
            elif breakout_long:
                signals[i] = 0.25
                position = 1
            elif breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade short or breakdown below S3
            if fade_short or (close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade long or breakout above R3
            if fade_long or (close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>

#!/usr/bin/env python3
"""
6s Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (20 for volume)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Fade at R3/S3 (counter-trend)
        # Long fade: price touches S3 and bounces up with volume
        fade_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and vol_filter[i]
        # Short fade: price touches R3 and rejects down with volume
        fade_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and vol_filter[i]
        
        # Breakout at R4/S4 (trend continuation)
        # Long breakout: price breaks above R4 with volume
        breakout_long = (close[i] > r4_1d_aligned[i]) and vol_filter[i]
        # Short breakdown: price breaks below S4 with volume
        breakout_short = (close[i] < s4_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
            elif breakout_long:
                signals[i] = 0.25
                position = 1
            elif breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade short or breakdown below S3
            if fade_short or (close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade long or breakout above R3
            if fade_long or (close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (20 for volume)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Fade at R3/S3 (counter-trend)
        # Long fade: price touches S3 and bounces up with volume
        fade_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and vol_filter[i]
        # Short fade: price touches R3 and rejects down with volume
        fade_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and vol_filter[i]
        
        # Breakout at R4/S4 (trend continuation)
        # Long breakout: price breaks above R4 with volume
        breakout_long = (close[i] > r4_1d_aligned[i]) and vol_filter[i]
        # Short breakdown: price breaks below S4 with volume
        breakout_short = (close[i] < s4_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
            elif breakout_long:
                signals[i] = 0.25
                position = 1
            elif breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade short or breakdown below S3
            if fade_short or (close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade long or breakout above R3
            if fade_long or (close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (20 for volume)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Fade at R3/S3 (counter-trend)
        # Long fade: price touches S3 and bounces up with volume
        fade_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and vol_filter[i]
        # Short fade: price touches R3 and rejects down with volume
        fade_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and vol_filter[i]
        
        # Breakout at R4/S4 (trend continuation)
        # Long breakout: price breaks above R4 with volume
        breakout_long = (close[i] > r4_1d_aligned[i]) and vol_filter[i]
        # Short breakdown: price breaks below S4 with volume
        breakout_short = (close[i] < s4_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
            elif breakout_long:
                signals[i] = 0.25
                position = 1
            elif breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade short or breakdown below S3
            if fade_short or (close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade long or breakout above R3
            if fade_long or (close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (20 for volume)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Fade at R3/S3 (counter-trend)
        # Long fade: price touches S3 and bounces up with volume
        fade_long = (low[i] <= s3_1d_aligned[i]) and (close[i] > s3_1d_aligned[i]) and vol_filter[i]
        # Short fade: price touches R3 and rejects down with volume
        fade_short = (high[i] >= r3_1d_aligned[i]) and (close[i] < r3_1d_aligned[i]) and vol_filter[i]
        
        # Breakout at R4/S4 (trend continuation)
        # Long breakout: price breaks above R4 with volume
        breakout_long = (close[i] > r4_1d_aligned[i]) and vol_filter[i]
        # Short breakdown: price breaks below S4 with volume
        breakout_short = (close[i] < s4_1d_aligned[i]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
            elif breakout_long:
                signals[i] = 0.25
                position = 1
            elif breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade short or breakdown below S3
            if fade_short or (close[i] < s3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on fade long or breakout above R3
            if fade_long or (close[i] > r3_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4
- Calculate daily Camarilla pivot levels (R3, S3, R4, S4)
- Fade (counter-trend) at R3/S3 when price reverses with volume confirmation
- Breakout continuation at R4/S4 when price breaks with volume confirmation
- Exit: reversal signal or opposite signal
- Position size: 0.25 (25%)
- Target: 75-150 trades over 4 years (19-38/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14231_6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    pp = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    r4 = pp + (range_val * 1.1 / 2.0)
    r3 = pp + (range_val * 1.1 / 4.0)
    s3 = pp - (range_val * 1.1 / 4.0)
    s4 = pp - (range_val * 1.1 / 2.0)
    
    return r4, r3, s3, s4

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d, r3_1d, s3_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s