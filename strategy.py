#!/usr/bin/env python3
"""
Experiment #8967: 6h Donchian breakout + 1d pivot direction + volume confirmation.
Hypothesis: Donchian breakouts capture trends; 1d pivot levels filter direction; volume confirms institutional participation.
Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost. Works in bull (breakouts) and bear (filtered shorts).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_PERIOD = 1  # daily pivot
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_pivot(high, low, close):
    """Calculate pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot(high_1d, low_1d, close_1d)
    
    # Determine bias based on price vs pivot (1=above pivot bullish, -1=below pivot bearish)
    price_vs_pivot = np.where(close_1d > pivot, 1, 
                              np.where(close_1d < pivot, -1, 0))
    price_vs_pivot_aligned = align_htf_to_ltf(prices, df_1d, price_vs_pivot)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1d pivot
        bull_bias = price_vs_pivot_aligned[i] == 1   # price above pivot
        bear_bias = price_vs_pivot_aligned[i] == -1  # price below pivot
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
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
Experiment #8967: 6h Camarilla pivot + volume confirmation + ATR filter.
Hypothesis: Camarilla levels from 1d identify intraday support/resistance; fade at R3/S3, breakout at R4/S4 with volume confirmation.
Targets 75-200 total trades over 4 years (19-50/year). Works in bull (breakouts) and bear (fades).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # daily
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r4 = close + range_val * 1.1 / 2
    r3 = close + range_val * 1.1 / 4
    r2 = close + range_val * 1.1 / 6
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Determine zones: 1=above R4 (bullish breakout), -1=below S4 (bearish breakdown), 0=between
    price_vs_r4 = np.where(close_1d > r4, 1, 0)
    price_vs_s4 = np.where(close_1d < s4, -1, 0)
    # Combine: priority to breakout levels
    camarilla_signal = np.where(price_vs_r4 == 1, 1, np.where(price_vs_s4 == -1, -1, 0))
    camarilla_signal_aligned = align_htf_to_ltf(prices, df_1d, camarilla_signal)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(camarilla_signal_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 1d Camarilla
        bull_breakout = camarilla_signal_aligned[i] == 1   # price above R4
        bear_breakout = camarilla_signal_aligned[i] == -1  # price below S4
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_breakout and volume_confirmed
        short_entry = bear_breakout and volume_confirmed
        
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
Experiment #8967: 6h Elder Ray (Bull/Bear Power) + EMA filter + volume confirmation.
Hypothesis: Elder Ray measures bull/bear power via EMA(13); trade in direction of power with EMA(34) trend filter and volume confirmation.
Targets 75-200 total trades over 4 years (19-50/year). Works in bull (bull power > 0) and bear (bear power < 0).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_elder_ray_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_FAST = 13
ELDER_RAY_SLOW = 34
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.6
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

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
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=ELDER_RAY_FAST, adjust=False, min_periods=ELDER_RAY_FAST).mean().values
    ema_slow = close_series.ewm(span=ELDER_RAY_SLOW, adjust=False, min_periods=ELDER_RAY_SLOW).mean().values
    
    bull_power = high - ema_fast
    bear_power = low - ema_fast
    
    # EMA trend filter: above EMA(34) = bullish bias, below = bearish bias
    ema_trend = np.where(close > ema_slow, 1, np.where(close < ema_slow, -1, 0))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
        
        # Determine market conditions
        bull_power_pos = bull_power[i] > 0  # bulls in control
        bear_power_neg = bear_power[i] < 0  # bears in control
        bullish_trend = ema_trend[i] == 1   # price above EMA(34)
        bearish_trend = ema_trend[i] == -1  # price below EMA(34)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: trade in direction of power with trend filter
        long_entry = bull_power_pos and bullish_trend and volume_confirmed
        short_entry = bear_power_neg and bearish_trend and volume_confirmed
        
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
Experiment #8967: 6h Williams Alligator + EMA filter + volume confirmation.
Hypothesis: Alligator (SMAs 5,8,13) identifies trends; trade when aligned with EMA(21) and volume confirmation.
Targets 75-200 total trades over 4 years (19-50/year). Works in bull (jaws-teeth-lips aligned up) and bear (aligned down).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_alligator_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAWS = 13   # smoothed SMA(13)
ALLIGATOR_TEETH = 8   # smoothed SMA(8)
ALLIGATOR_LIPS = 5    # smoothed SMA(5)
EMA_FILTER = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs with smoothing (like SMMA)
    close_series = pd.Series(close)
    # Jaws: SMA(13) smoothed
    jaws = close_series.rolling(window=ALLIGATOR_JAWS, min_periods=ALLIGATOR_JAWS).mean()
    jaws = jaws.rolling(window=3, min_periods=3).mean()  # additional smoothing
    # Teeth: SMA(8) smoothed
    teeth = close_series.rolling(window=ALLIGATOR_TEETH, min_periods=ALLIGATOR_TEETH).mean()
    teeth = teeth.rolling(window=3, min_periods=3).mean()
    # Lips: SMA(5) smoothed
    lips = close_series.rolling(window=ALLIGATOR_LIPS, min_periods=ALLIGATOR_LIPS).mean()
    lips = lips.rolling(window=3, min_periods=3).mean()
    
    jaws_val = jaws.values
    teeth_val = teeth.values
    lips_val = lips.values
    
    # Alligator alignment: 1=bullish (lips > teeth > jaws), -1=bearish (lips < teeth < jaws), 0=otherwise
    bullish_align = (lips_val > teeth_val) & (teeth_val > jaws_val)
    bearish_align = (lips_val < teeth_val) & (teeth_val < jaws_val)
    alligator_signal = np.where(bullish_align, 1, np.where(bearish_align, -1, 0))
    
    # EMA trend filter
    ema_filter = close_series.ewm(span=EMA_FILTER, adjust=False, min_periods=EMA_FILTER).mean().values
    ema_trend = np.where(close > ema_filter, 1, np.where(close < ema_filter, -1, 0))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAWS, EMA_FILTER, VOLUME_MA_PERIOD, ATR_PERIOD) + 2  # +2 for smoothing
    
    for i in range(start, n):
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
        
        # Determine market conditions
        bullish_alligator = alligator_signal[i] == 1   # jaws-teeth-lips aligned up
        bearish_alligator = alligator_signal[i] == -1  # aligned down
        bullish_trend = ema_trend[i] == 1              # price above EMA(21)
        bearish_trend = ema_trend[i] == -1             # price below EMA(21)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: need both Alligator alignment and EMA trend
        long_entry = bullish_alligator and bullish_trend and volume_confirmed
        short_entry = bearish_alligator and bearish_trend and volume_confirmed
        
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
Experiment #8967: 6h ADX + DiNapoli levels + volume confirmation.
Hypothesis: ADX > 25 indicates trending market; DiNapoli levels (0.382, 0.618) provide entry points in trend direction with volume confirmation.
Targets 75-200 total trades over 4 years (19-50/year). Works in bull (buy dips) and bear (sell rallies).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_adx_dinapoli_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.6
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_dinapoli(high, low, close):
    """Calculate DiNapoli levels: 0.382 and 0.618 retracements of recent swing"""
    # Find recent swing high and low over lookback period
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=1).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=1).min().values
    
    # Calculate the range
    swing_range = highest_high - lowest_low
    
    # DiNapoli levels: 0.382 and 0.618 retracements from low
    level_382 = lowest_low + swing_range * 0.382
    level_618 = lowest_low + swing_range * 0.618
    
    return level_382, level_618, highest_high, lowest_low

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX calculation
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # DiNapoli levels
    level_382, level_618, highest_high, lowest_low = calculate_dinapoli(high, low, close)
    
    # Determine trend direction from ADX and DI crossover
    # 1=uptrend (PDI > MDI), -1=downtrend (MDI > PDI), 0=no trend
    trend_direction = np.where(plus_di > minus_di, 1, np.where(minus_di > plus_di, -1, 0))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
        
        # Determine market conditions
        strong_trend = adx[i] > ADX_THRESHOLD  # ADX > 25 indicates strong trend
        uptrend = trend_direction[i] == 1      # PDI > MDI
        downtrend = trend_direction[i] == -1   # MDI > PDI
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: buy near 0.382 in uptrend, sell near 0.618 in downtrend
        long_entry = strong_trend and uptrend and volume_confirmed and (close[i] <= level_382[i])
        short_entry = strong_trend and downtrend and volume_confirmed and (close[i] >= level_618[i])
        
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
Experiment #8967: 6h Donchian breakout + 1d ADX filter + volume confirmation.
Hypothesis: Donchian breakouts capture trends; 1d ADX > 25 filters for strong trends; volume confirms participation.
Targets 75-200 total trades over 4 years (19-50/year). Works in bull (breakouts) and bear (filtered shorts).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8967_6h_donchian20_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1