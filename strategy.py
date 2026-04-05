#!/usr/bin/env python3
"""
Experiment #10672: 12h Donchian Breakout + Daily Trend + Volume Spike
Hypothesis: 12-hour Donchian(20) breakouts aligned with daily EMA50 trend and volume spikes provide
robust trend-following signals. Works in bull markets (breakouts above daily EMA) and bear markets
(breakouts below daily EMA). Volume filters reduce false breakouts. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10672_12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 12h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, DAILY_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of daily trend with volume
        long_entry = bullish_breakout and above_daily_ema and volume_spike
        short_entry = bearish_breakout and below_daily_ema and volume_spike
        
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
Experiment #10672: 12h Camarilla Pivot + Daily Trend + Volume Spike
Hypothesis: 12-hour Camarilla pivot level touches aligned with daily EMA50 trend and volume spikes
provide high-probability mean-reversion entries in ranging markets and trend continuation in trending.
Works in both bull and bear markets by adapting to price action at key institutional levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10672_12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_MULTIPLIER = 1.1
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading"""
    # Camarilla levels based on previous day's range
    range_val = high - low
    # Resistance levels
    r4 = close + range_val * CAMARILLA_MULTIPLIER * 1.500
    r3 = close + range_val * CAMARILLA_MULTIPLIER * 1.250
    r2 = close + range_val * CAMARILLA_MULTIPLIER * 1.166
    r1 = close + range_val * CAMARILLA_MULTIPLIER * 1.083
    # Support levels
    s1 = close - range_val * CAMARILLA_MULTIPLIER * 1.083
    s2 = close - range_val * CAMARILLA_MULTIPLIER * 1.166
    s3 = close - range_val * CAMARILLA_MULTIPLIER * 1.250
    s4 = close - range_val * CAMARILLA_MULTIPLIER * 1.500
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter and pivot calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 12h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivots from previous daily bar
    # Shift daily data by 1 to use previous day's OHLC for current 12h bar
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close_vals = df_daily['close'].values
    
    # Calculate pivots for each day, then align to 12h
    r1_daily, r2_daily, r3_daily, r4_daily, s1_daily, s2_daily, s3_daily, s4_daily = calculate_camarilla_pivots(
        daily_high, daily_low, daily_close_vals
    )
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DAILY_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Camarilla level touches (using previous day's levels)
        # Long when price touches S1 or S2 in uptrend, or S3/S4 in strong downtrend (mean reversion)
        # Short when price touches R1 or R2 in downtrend, or R3/R4 in strong uptrend
        touch_s1 = abs(close[i] - s1_aligned[i]) < (0.001 * close[i]) if not np.isnan(s1_aligned[i]) else False
        touch_s2 = abs(close[i] - s2_aligned[i]) < (0.001 * close[i]) if not np.isnan(s2_aligned[i]) else False
        touch_s3 = abs(close[i] - s3_aligned[i]) < (0.001 * close[i]) if not np.isnan(s3_aligned[i]) else False
        touch_s4 = abs(close[i] - s4_aligned[i]) < (0.001 * close[i]) if not np.isnan(s4_aligned[i]) else False
        touch_r1 = abs(close[i] - r1_aligned[i]) < (0.001 * close[i]) if not np.isnan(r1_aligned[i]) else False
        touch_r2 = abs(close[i] - r2_aligned[i]) < (0.001 * close[i]) if not np.isnan(r2_aligned[i]) else False
        touch_r3 = abs(close[i] - r3_aligned[i]) < (0.001 * close[i]) if not np.isnan(r3_aligned[i]) else False
        touch_r4 = abs(close[i] - r4_aligned[i]) < (0.001 * close[i]) if not np.isnan(r4_aligned[i]) else False
        
        # Entry conditions
        # Long: touch support in uptrend OR touch strong support in downtrend (mean reversion)
        long_entry = ((touch_s1 or touch_s2) and above_daily_ema and volume_spike) or \
                     ((touch_s3 or touch_s4) and below_daily_ema and volume_spike)
        # Short: touch resistance in downtrend OR touch strong resistance in uptrend (mean reversion)
        short_entry = ((touch_r1 or touch_r2) and below_daily_ema and volume_spike) or \
                      ((touch_r3 or touch_r4) and above_daily_ema and volume_spike)
        
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
Experiment #10672: 12h Trix + Daily Trend + Volume Spike
Hypothesis: 12-hour Trix (triple exponential average) crossovers aligned with daily EMA50 trend
and volume spikes provide reliable momentum signals. Works in both bull and bear markets
by capturing momentum shifts at key turning points. Volume filters reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10672_12h_trix_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
TRIX_PERIOD = 12
TRIX_SIGNAL_PERIOD = 9
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_trix(close, period):
    """Calculate TRIX (Triple Exponential Average)"""
    # First EMA
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    # Second EMA
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    # Third EMA
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # TRIX = ((EMA3 today - EMA3 yesterday) / EMA3 yesterday) * 100
    trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    return trix.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 12h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX indicator
    trix = calculate_trix(close, TRIX_PERIOD)
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=TRIX_SIGNAL_PERIOD, adjust=False, min_periods=TRIX_SIGNAL_PERIOD).mean().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TRIX_PERIOD + TRIX_SIGNAL_PERIOD, DAILY_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # TRIX crossover signals
        trix_cross_above = (trix[i] > trix_signal[i]) and (trix[i-1] <= trix_signal[i-1]) if i > 0 else False
        trix_cross_below = (trix[i] < trix_signal[i]) and (trix[i-1] >= trix_signal[i-1]) if i > 0 else False
        
        # Entry conditions: TRIX crossover in direction of daily trend with volume
        long_entry = trix_cross_above and above_daily_ema and volume_spike
        short_entry = trix_cross_below and below_daily_ema and volume_spike
        
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
Experiment #10672: 12h Vortex Indicator + Daily Trend + Volume Spike
Hypothesis: 12-hour Vortex Indicator (VI) crossovers aligned with daily EMA50 trend
and volume spikes provide strong trend initiation signals. Works in both bull and bear
markets by detecting the emergence of new trends. Volume filters reduce false breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10672_12h_vortex_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
VORTEX_PERIOD = 14
DAILY_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_vortex_indicator(high, low, close, period):
    """Calculate Vortex Indicator (VI)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Vertical Movement
    vm_plus = np.abs(high - np.roll(low, 1))   # +VM
    vm_minus = np.abs(low - np.roll(high, 1))  # -VM
    
    # Sum over period
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    sum_vm_plus = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    sum_vm_minus = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI+ and VI-
    vi_plus = sum_vm_plus / sum_tr
    vi_minus = sum_vm_minus / sum_tr
    
    return vi_plus, vi_minus

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load daily data ONCE before loop for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend direction
    daily_close = df_daily['close'].values
    daily_ema = calculate_ema(daily_close, DAILY_EMA_PERIOD)
    
    # Align daily EMA to 12h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, daily_ema)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Vortex Indicator
    vi_plus, vi_minus = calculate_vortex_indicator(high, low, close, VORTEX_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VORTEX_PERIOD, DAILY_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(daily_ema_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below daily EMA
        above_daily_ema = close[i] > daily_ema_aligned[i]
        below_daily_ema = close[i] < daily_ema_aligned[i]
        
        # Vortex crossover signals
        vi_cross_above = (vi_plus[i] > vi_minus[i]) and (vi_plus[i-1] <= vi_minus[i-1]) if i > 0 else False
        vi_cross_below = (vi_plus[i] < vi_minus[i]) and (vi_plus[i-1] >= vi_minus[i-1]) if i > 0 else False
        
        # Entry conditions: VI crossover in direction of daily trend with volume
        long_entry = vi_cross_above and above_daily_ema and volume_spike
        short_entry = vi_cross_below and below_daily_ema and volume_spike
        
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