#!/usr/bin/env python3
"""
Experiment #8654: 1h trend following with 4h trend filter and 1d volume confirmation
Hypothesis: 1h timeframe with strict entry conditions (4h trend + 1d volume spike) will generate 60-150 trades over 4 years.
Uses 4h for trend direction (Hull Moving Average) and 1d for volume confirmation (volume > 1.5x 20-day average).
1h timeframe used only for entry timing precision with HMA crossovers.
Targets 15-37 trades/year to minimize fee drag while maintaining statistical validity.
Includes session filter (08-20 UTC) to avoid low-volume Asian session.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8654_1h_hma4h_vol1d_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend filter
    close_4h = df_4h['close'].values
    hma_fast_4h = calculate_hma(close_4h, HMA_FAST)
    hma_slow_4h = calculate_hma(close_4h, HMA_SLOW)
    # Trend: 1 if fast > slow (bullish), -1 if fast < slow (bearish)
    trend_4h = np.where(hma_fast_4h > hma_slow_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d volume MA for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h HMA for entry signals
    hma_fast_1h = calculate_hma(close, HMA_FAST)
    hma_slow_1h = calculate_hma(close, HMA_SLOW)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from 4h HMA
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # HMA crossover signals on 1h
        hma_cross_up = hma_fast_1h[i] > hma_slow_1h[i] and hma_fast_1h[i-1] <= hma_slow_1h[i-1]
        hma_cross_down = hma_fast_1h[i] < hma_slow_1h[i] and hma_fast_1h[i-1] >= hma_slow_1h[i-1]
        
        # Entry conditions
        long_entry = bull_trend and volume_confirmed and hma_cross_up
        short_entry = bear_trend and volume_confirmed and hma_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8654: 1h trend following with 4h trend filter and 1d volume confirmation
Hypothesis: 1h timeframe with strict entry conditions (4h trend + 1d volume spike) will generate 60-150 trades over 4 years.
Uses 4h for trend direction (Hull Moving Average) and 1d for volume confirmation (volume > 1.5x 20-day average).
1h timeframe used only for entry timing precision with HMA crossovers.
Targets 15-37 trades/year to minimize fee drag while maintaining statistical validity.
Includes session filter (08-20 UTC) to avoid low-volume Asian session.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8654_1h_hma4h_vol1d_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend filter
    close_4h = df_4h['close'].values
    hma_fast_4h = calculate_hma(close_4h, HMA_FAST)
    hma_slow_4h = calculate_hma(close_4h, HMA_SLOW)
    # Trend: 1 if fast > slow (bullish), -1 if fast < slow (bearish)
    trend_4h = np.where(hma_fast_4h > hma_slow_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d volume MA for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h HMA for entry signals
    hma_fast_1h = calculate_hma(close, HMA_FAST)
    hma_slow_1h = calculate_hma(close, HMA_SLOW)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from 4h HMA
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # HMA crossover signals on 1h
        hma_cross_up = hma_fast_1h[i] > hma_slow_1h[i] and hma_fast_1h[i-1] <= hma_slow_1h[i-1]
        hma_cross_down = hma_fast_1h[i] < hma_slow_1h[i] and hma_fast_1h[i-1] >= hma_slow_1h[i-1]
        
        # Entry conditions
        long_entry = bull_trend and volume_confirmed and hma_cross_up
        short_entry = bear_trend and volume_confirmed and hma_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8654: 1h trend following with 4h trend filter and 1d volume confirmation
Hypothesis: 1h timeframe with strict entry conditions (4h trend + 1d volume spike) will generate 60-150 trades over 4 years.
Uses 4h for trend direction (Hull Moving Average) and 1d for volume confirmation (volume > 1.5x 20-day average).
1h timeframe used only for entry timing precision with HMA crossovers.
Targets 15-37 trades/year to minimize fee drag while maintaining statistical validity.
Includes session filter (08-20 UTC) to avoid low-volume Asian session.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8654_1h_hma4h_vol1d_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend filter
    close_4h = df_4h['close'].values
    hma_fast_4h = calculate_hma(close_4h, HMA_FAST)
    hma_slow_4h = calculate_hma(close_4h, HMA_SLOW)
    # Trend: 1 if fast > slow (bullish), -1 if fast < slow (bearish)
    trend_4h = np.where(hma_fast_4h > hma_slow_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d volume MA for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h HMA for entry signals
    hma_fast_1h = calculate_hma(close, HMA_FAST)
    hma_slow_1h = calculate_hma(close, HMA_SLOW)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from 4h HMA
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # HMA crossover signals on 1h
        hma_cross_up = hma_fast_1h[i] > hma_slow_1h[i] and hma_fast_1h[i-1] <= hma_slow_1h[i-1]
        hma_cross_down = hma_fast_1h[i] < hma_slow_1h[i] and hma_fast_1h[i-1] >= hma_slow_1h[i-1]
        
        # Entry conditions
        long_entry = bull_trend and volume_confirmed and hma_cross_up
        short_entry = bear_trend and volume_confirmed and hma_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #8654: 1h trend following with 4h trend filter and 1d volume confirmation
Hypothesis: 1h timeframe with strict entry conditions (4h trend + 1d volume spike) will generate 60-150 trades over 4 years.
Uses 4h for trend direction (Hull Moving Average) and 1d for volume confirmation (volume > 1.5x 20-day average).
1h timeframe used only for entry timing precision with HMA crossovers.
Targets 15-37 trades/year to minimize fee drag while maintaining statistical validity.
Includes session filter (08-20 UTC) to avoid low-volume Asian session.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8654_1h_hma4h_vol1d_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SESSION_START_HOUR = 8
SESSION_END_HOUR = 20

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend filter
    close_4h = df_4h['close'].values
    hma_fast_4h = calculate_hma(close_4h, HMA_FAST)
    hma_slow_4h = calculate_hma(close_4h, HMA_SLOW)
    # Trend: 1 if fast > slow (bullish), -1 if fast < slow (bearish)
    trend_4h = np.where(hma_fast_4h > hma_slow_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d volume MA for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h HMA for entry signals
    hma_fast_1h = calculate_hma(close, HMA_FAST)
    hma_slow_1h = calculate_hma(close, HMA_SLOW)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(HMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START_HOUR or hour > SESSION_END_HOUR:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from 4h HMA
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD)
        
        # HMA crossover signals on 1h
        hma_cross_up = hma_fast_1h[i] > hma_slow_1h[i] and hma_fast_1h[i-1] <= hma_slow_1h[i-1]
        hma_cross_down = hma_fast_1h[i] < hma_slow_1h[i] and hma_fast_1h[i-1] >= hma_slow_1h[i-1]
        
        # Entry conditions
        long_entry = bull_trend and volume_confirmed and hma_cross_up
        short_entry = bear_trend and volume_confirmed and hma_cross_down
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

--- END OF FILE ---