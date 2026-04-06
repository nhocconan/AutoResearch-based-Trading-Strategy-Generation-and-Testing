#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h EMA for trend direction (once before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d Bollinger Bands for mean reversion zones (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d.rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Precompute volume spike (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA and Bollinger Bands
    
    for i in range(start, n):
        # Skip if required data not available
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches upper Bollinger Band or stoploss hit
            if (close[i] >= upper_bb_aligned[i] or
                close[i] < entry_price - 2.0 * (high[i] - low[i])):  # Simple volatility stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower Bollinger Band or stoploss hit
            if (close[i] <= lower_bb_aligned[i] or
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: price near Bollinger Bands + volume spike + 4h trend filter
            vol_spike = volume[i] > vol_ma[i] * 2.0 if not np.isnan(vol_ma[i]) else False
            
            near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.01  # Within 1% of lower BB
            near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.99  # Within 1% of upper BB
            
            # 4h trend filter: only long in uptrend, short in downtrend
            uptrend = close[i] > ema_4h_aligned[i]
            downtrend = close[i] < ema_4h_aligned[i]
            
            if near_lower_bb and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif near_upper_bb and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
1h Mean Reversion with 4h/1d Trend Filter and Volume Spike
Hypothesis: In ranging markets (common in 2025), price reverts to mean after volatility spikes.
Trades only during high-liquidity hours (08-20 UTC) and in direction of higher timeframe trend.
Uses 4h EMA for trend direction, 1d Bollinger Bands for mean reversion zones, and volume spike for entry timing.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4htrend_1dbb_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].