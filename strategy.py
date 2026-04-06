#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using 1-day Elder Ray (Bull/Bear Power) with 1-week regime filter.
# Elder Ray measures bullish/bearish power: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Uses 1-week EMA to determine regime: bullish when price > weekly EMA, bearish when price < weekly EMA.
# In bullish regime: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In bearish regime: short when Bear Power < 0 and falling, long when Bull Power > 0 and rising.
# Volume confirmation: require volume > 1.5x 8-period MA.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (trend following with Elder Ray) and bear (mean reversion at extremes) markets.

name = "exp_13767_6h_elder_ray_1w_regime_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_EMA_PERIOD = 13
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, ELDER_EMA_PERIOD)
    
    # Calculate Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_1d
    bear_power = ema_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 1w data for regime filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_EMA_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filter: bullish when price > weekly EMA, bearish when price < weekly EMA
        bullish_regime = close[i] > ema_1w_aligned[i]
        bearish_regime = close[i] < ema_1w_aligned[i]
        
        # Elder Ray signals with slope (change from previous period)
        bull_power_curr = bull_power_aligned[i]
        bear_power_curr = bear_power_aligned[i]
        bull_power_prev = bull_power_aligned[i-1]
        bear_power_prev = bear_power_aligned[i-1]
        
        bull_power_rising = bull_power_curr > bull_power_prev
        bear_power_falling = bear_power_curr < bear_power_prev
        
        # Long conditions: Bull Power > 0 and rising (bullish momentum)
        long_cond = bull_power_curr > 0 and bull_power_rising and volume_ok
        # Short conditions: Bear Power < 0 and falling (bearish momentum)
        short_cond = bear_power_curr < 0 and bear_power_falling and volume_ok
        
        # Regime adjustment: in bullish regime, favor longs; in bearish regime, favor shorts
        if bullish_regime:
            # In bullish regime: take longs, avoid shorts unless strong bearish signal
            if long_cond:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_cond and bear_power_curr < -0.5:  # stronger short signal needed in bullish regime
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        elif bearish_regime:
            # In bearish regime: take shorts, avoid longs unless strong bullish signal
            if short_cond:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_cond and bull_power_curr > 0.5:  # stronger long signal needed in bearish regime
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        else:
            # Exactly at weekly EMA (rare)
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using 1-day Elder Ray (Bull/Bear Power) with 1-week regime filter.
# Elder Ray measures bullish/bearish power: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Uses 1-week EMA to determine regime: bullish when price > weekly EMA, bearish when price < weekly EMA.
# In bullish regime: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In bearish regime: short when Bear Power < 0 and falling, long when Bull Power > 0 and rising.
# Volume confirmation: require volume > 1.5x 8-period MA.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull (trend following with Elder Ray) and bear (mean reversion at extremes) markets.

name = "exp_13767_6h_elder_ray_1w_regime_vol"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_EMA_PERIOD = 13
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 8
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray calculation ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, ELDER_EMA_PERIOD)
    
    # Calculate Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_1d
    bear_power = ema_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 1w data for regime filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6h data for entry timing and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_EMA_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filter: bullish when price > weekly EMA, bearish when price < weekly EMA
        bullish_regime = close[i] > ema_1w_aligned[i]
        bearish_regime = close[i] < ema_1w_aligned[i]
        
        # Elder Ray signals with slope (change from previous period)
        bull_power_curr = bull_power_aligned[i]
        bear_power_curr = bear_power_aligned[i]
        bull_power_prev = bull_power_aligned[i-1]
        bear_power_prev = bear_power_aligned[i-1]
        
        bull_power_rising = bull_power_curr > bull_power_prev
        bear_power_falling = bear_power_curr < bear_power_prev
        
        # Long conditions: Bull Power > 0 and rising (bullish momentum)
        long_cond = bull_power_curr > 0 and bull_power_rising and volume_ok
        # Short conditions: Bear Power < 0 and falling (bearish momentum)
        short_cond = bear_power_curr < 0 and bear_power_falling and volume_ok
        
        # Regime adjustment: in bullish regime, favor longs; in bearish regime, favor shorts
        if bullish_regime:
            # In bullish regime: take longs, avoid shorts unless strong bearish signal
            if long_cond:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_cond and bear_power_curr < -0.5:  # stronger short signal needed in bullish regime
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        elif bearish_regime:
            # In bearish regime: take shorts, avoid longs unless strong bullish signal
            if short_cond:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_cond and bull_power_curr > 0.5:  # stronger long signal needed in bearish regime
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        else:
            # Exactly at weekly EMA (rare)
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals