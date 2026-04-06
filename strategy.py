#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly volatility regime filter and volume confirmation.
# Donchian breakouts capture momentum in both bull and bear markets.
# Weekly ATR-based volatility regime filter avoids whipsaws in high volatility (bear market) and
# enters only when volatility is expanding from a low base (breakout from consolidation).
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).

name = "exp_13595_6h_donchian20_1w_vol_regime_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLATILITY_LOOKBACK = 20  # for weekly ATR
VOLUME_CONFIRM = 20  # volume MA period
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

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
    
    # Load weekly data for volatility regime filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, VOLATILITY_LOOKBACK)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    # Volatility regime: current ATR > MA(ATR) indicates expanding volatility
    vol_regime = atr_1w > atr_ma_1w
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_CONFIRM, min_periods=VOLUME_CONFIRM).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLATILITY_LOOKBACK, VOLUME_CONFIRM, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_regime_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above upper channel
        breakdown_down = close[i] < donchian_lower[i-1]  # break below lower channel
        
        # Generate signals
        if position == 0:
            if volume_ok and vol_regime_aligned[i] and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and vol_regime_aligned[i] and breakdown_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or stop loss
            if close[i] < donchian_lower[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or stop loss
            if close[i] > donchian_upper[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly volatility regime filter and volume confirmation.
# Donchian breakouts capture momentum in both bull and bear markets.
# Weekly ATR-based volatility regime filter avoids whipsaws in high volatility (bear market) and
# enters only when volatility is expanding from a low base (breakout from consolidation).
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).

name = "exp_13595_6h_donchian20_1w_vol_regime_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLATILITY_LOOKBACK = 20  # for weekly ATR
VOLUME_CONFIRM = 20  # volume MA period
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

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
    
    # Load weekly data for volatility regime filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, VOLATILITY_LOOKBACK)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    # Volatility regime: current ATR > MA(ATR) indicates expanding volatility
    vol_regime = atr_1w > atr_ma_1w
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_CONFIRM, min_periods=VOLUME_CONFIRM).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLATILITY_LOOKBACK, VOLUME_CONFIRM, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_regime_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above upper channel
        breakdown_down = close[i] < donchian_lower[i-1]  # break below lower channel
        
        # Generate signals
        if position == 0:
            if volume_ok and vol_regime_aligned[i] and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and vol_regime_aligned[i] and breakdown_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or stop loss
            if close[i] < donchian_lower[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or stop loss
            if close[i] > donchian_upper[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly volatility regime filter and volume confirmation.
# Donchian breakouts capture momentum in both bull and bear markets.
# Weekly ATR-based volatility regime filter avoids whipsaws in high volatility (bear market) and
# enters only when volatility is expanding from a low base (breakout from consolidation).
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).

name = "exp_13595_6h_donchian20_1w_vol_regime_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLATILITY_LOOKBACK = 20  # for weekly ATR
VOLUME_CONFIRM = 20  # volume MA period
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

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
    
    # Load weekly data for volatility regime filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, VOLATILITY_LOOKBACK)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    # Volatility regime: current ATR > MA(ATR) indicates expanding volatility
    vol_regime = atr_1w > atr_ma_1w
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_CONFIRM, min_periods=VOLUME_CONFIRM).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLATILITY_LOOKBACK, VOLUME_CONFIRM, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_regime_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above upper channel
        breakdown_down = close[i] < donchian_lower[i-1]  # break below lower channel
        
        # Generate signals
        if position == 0:
            if volume_ok and vol_regime_aligned[i] and breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and vol_regime_aligned[i] and breakdown_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or stop loss
            if close[i] < donchian_lower[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or stop loss
            if close[i] > donchian_upper[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

--- END ---