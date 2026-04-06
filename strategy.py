#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20] = np.mean(close_weekly[:21])
        for i in range(21, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or price below/above weekly EMA
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price crosses below weekly EMA
            if close[i] < donchian_low[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price crosses above weekly EMA
            if close[i] > donchian_high[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA filter
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            ema_filter_long = close[i] > ema_weekly_aligned[i]  # Above weekly EMA for long
            ema_filter_short = close[i] < ema_weekly_aligned[i]  # Below weekly EMA for short
            
            if bull_breakout and volume_filter and ema_filter_long:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and ema_filter_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1D Donchian 20 Breakout with Volume Confirmation and Weekly EMA Filter
Hypothesis: Daily Donchian breakouts capture multi-day trends. Volume ensures breakout strength,
while weekly EMA filter aligns with major trend. Designed for 30-100 trades over 4 years to
minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_weekly_ema_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = np.full_like(close_weekly, np.nan)
    if len(close_weekly) >= 21:
        ema_weekly[20]