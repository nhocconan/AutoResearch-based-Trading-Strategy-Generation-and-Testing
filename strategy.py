#!/usr/bin/env python3
"""
Experiment #9767: 6h 40-period Moving Average Crossover with Volume Confirmation and Regime Filter.
Hypothesis: A 40-period MA crossover (fast: 20, slow: 40) on 6h timeframe, combined with volume spikes 
and regime filtering (ADX > 20 for trend confirmation), provides robust trend-following signals that 
work in both bull and bear markets. The 40-period setting avoids whipsaws while capturing major trends.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9767_6h_ma_crossover_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
FAST_MA_PERIOD = 20
SLOW_MA_PERIOD = 40
VOLUME_SPIKE_MULTIPLIER = 1.5
ADX_PERIOD = 14
ADX_THRESHOLD = 20
SIGNAL_SIZE = 0.30
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    return np.maximum(np.maximum(tr1, tr2), tr3)

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
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
    
    # EMAs for crossover
    ema_fast = calculate_ema(close, FAST_MA_PERIOD)
    ema_slow = calculate_ema(close, SLOW_MA_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(FAST_MA_PERIOD, SLOW_MA_PERIOD, ADX_PERIOD, ATR_PERIOD, 20) + 1
    
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: ADX > 20 for trend confirmation
        trending = adx[i] >= ADX_THRESHOLD
        
        # Crossover signals
        golden_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        death_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Entry conditions: crossover + volume + trend
        long_entry = golden_cross and volume_spike and trending
        short_entry = death_cross and volume_spike and trending
        
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
Experiment #9767: 6h Donchian Breakout with Volume Confirmation and Volatility Filter.
Hypothesis: Donchian channel breakouts (20-period) on 6h timeframe, combined with volume spikes 
and volatility filtering (ATR ratio < 1.2 for breakout validation), provide high-probability 
trend continuation signals. Works in bull (upper band breaks) and bear (lower band breaks) 
markets by capturing momentum after consolidation. Targets 80-160 total trades over 4 years 
(20-40/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9767_6h_donchian_breakout_volume_volatility_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
ATR_RATIO_THRESHOLD = 1.2
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for volatility and risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio (current ATR vs average ATR) for volatility filtering
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ATR_PERIOD, 50) + 1
    
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Volatility filter: ATR ratio < threshold for breakout validation
        low_volatility = not np.isnan(atr_ratio[i]) and atr_ratio[i] < ATR_RATIO_THRESHOLD
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Entry conditions: breakout + volume + volatility filter
        long_entry = breakout_up and volume_spike and low_volatility
        short_entry = breakout_down and volume_spike and low_volatility
        
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
Experiment #9767: 6h ADX Trend Strength with Volatility-Adjusted Position Sizing.
Hypothesis: ADX trend strength indicator on 6h timeframe, combined with volatility-adjusted 
position sizing (inverse ATR scaling) and volume confirmation, provides adaptive trend 
following that works in both bull and bear markets. Higher ADX = stronger trend = larger 
position (up to 0.35). Targets 90-180 total trades over 4 years (22-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9767_6h_adx_trend_strength_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_ENTRY_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 1.6
SIGNAL_SIZE_BASE = 0.25
SIGNAL_SIZE_MAX = 0.35
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    return np.maximum(np.maximum(tr1, tr2), tr3)

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
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
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for volatility and risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ATR_PERIOD, 20) + 1
    
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: ADX > threshold for trend confirmation
        trending = adx[i] >= ADX_ENTRY_THRESHOLD
        
        # Determine signal size based on ADX strength (normalized 0-100)
        if trending and volume_spike:
            # Scale position size from base to max based on ADX level
            adx_normalized = min(adx[i] / 50.0, 1.0)  # Cap at ADX=50 for normalization
            signal_size = SIGNAL_SIZE_BASE + (SIGNAL_SIZE_MAX - SIGNAL_SIZE_BASE) * adx_normalized
            
            # Determine trend direction using price vs moving average
            ma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
            if not np.isnan(ma_50[i]):
                if close[i] > ma_50[i]:  # Uptrend
                    if position <= 0:  # Reverse or new long
                        signals[i] = signal_size
                        position = 1
                        entry_price = close[i]
                        stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                    else:
                        signals[i] = signal_size
                else:  # Downtrend
                    if position >= 0:  # Reverse or new short
                        signals[i] = -signal_size
                        position = -1
                        entry_price = close[i]
                        stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                    else:
                        signals[i] = -signal_size
            else:
                signals[i] = 0.0
        else:
            # No trend or no volume: flat or maintain
            if position == 1:
                signals[i] = SIGNAL_SIZE_BASE
            elif position == -1:
                signals[i] = -SIGNAL_SIZE_BASE
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
Experiment #9767: 6h Volume-Weighted Average Price (VWAP) Deviation with Mean Reversion.
Hypothesis: Price deviations from VWAP on 6h timeframe, combined with volume spikes 
and Bollinger Band extremes, provide high-probability mean reversion signals. 
Works in ranging markets (common in crypto) and during trend pullbacks. 
Targets 70-140 total trades over 4 years (17-35/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9767_6h_vwap_deviation_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VWAP_WINDOW = 24  # 24 periods = 6 days on 6h chart
BB_PERIOD = 20
BB_STD_DEV = 2.0
VOLUME_SPIKE_MULTIPLIER = 1.7
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_vwap(high, low, close, volume, window):
    """Calculate Volume-Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    vwap_numerator = pd.Series(typical_price * volume).rolling(window=window, min_periods=window).sum().values
    vwap_denominator = pd.Series(volume).rolling(window=window, min_periods=window).sum().values
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    return vwap

def calculate_bollinger_bands(close, period, std_dev):
    """Calculate Bollinger Bands"""
    ma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper, lower, ma

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
    
    # VWAP
    vwap = calculate_vwap(high, low, close, volume, VWAP_WINDOW)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, BB_PERIOD, BB_STD_DEV)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VWAP_WINDOW, BB_PERIOD, ATR_PERIOD, 20) + 1
    
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion conditions: price at Bollinger Band extreme + volume spike
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # Entry conditions: mean reversion at extremes + volume
        long_entry = at_bb_lower and volume_spike
        short_entry = at_bb_upper and volume_spike
        
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