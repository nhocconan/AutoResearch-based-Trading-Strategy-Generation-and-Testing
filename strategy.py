#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_ath_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with pivot levels as key levels.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_1w_pivot_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 20
EMA_TREND = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate pivot points (using previous period's high/low/close)
    # Shift by 1 to use only completed periods
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Entry conditions: breakout of pivot levels with trend alignment
        long_entry = (
            weekly_uptrend and          # weekly uptrend
            close[i] > r1[i]            # break above R1
        )
        
        short_entry = (
            weekly_downtrend and        # weekly downtrend
            close[i] < s1[i]            # break below S1
        )
        
        # Exit conditions: return to pivot level
        long_exit = close[i] < pivot[i]
        short_exit = close[i] > pivot[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

#!/usr/bin/env python3
"""
exp_7495_6d_1w_pivot_v1
Hypothesis: 6d (6-hour) pivot breakout with weekly trend filter. 
Enter long when price breaks above R1 with weekly uptrend (price > weekly EMA50), 
enter short when price breaks below S1 with weekly downtrend (price < weekly EMA50).
Use weekly EMA50 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12