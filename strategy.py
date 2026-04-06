#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Donchian breakout with weekly EMA200 filter and volume confirmation.
# Long when price breaks above 6h Donchian(20) high with weekly EMA200 filter (price > EMA200) and volume > 1.5x average.
# Short when price breaks below 6h Donchian(20) low with weekly EMA200 filter (price < EMA200) and volume > 1.5x average.
# Exit on opposite Donchian break or trailing stop at 2x ATR.
# Weekly EMA200 provides trend filter that works in both bull and bear markets.
# Donchian breakouts capture momentum, volume confirms strength, and weekly filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13887_6h_donchian20_weekly_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load weekly data for EMA200 filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200
    close_weekly = df_weekly['close'].values
    ema200_weekly = calculate_ema(close_weekly, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA200 to 6h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # 6h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for trailing stop
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema200_weekly_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check trailing stop
        if position == 1:  # long position
            # Update stop price to trail behind
            stop_price = max(stop_price, high[i] - (ATR_TRAIL_MULTIPLIER * atr[i]))
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Update stop price to trail behind
            stop_price = min(stop_price, low[i] + (ATR_TRAIL_MULTIPLIER * atr[i]))
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Weekly EMA200 trend filter
        price_above_ema = close[i] > ema200_weekly_aligned[i]
        price_below_ema = close[i] < ema200_weekly_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and price_above_ema and close[i] > donchian_upper[i]
        short_signal = volume_ok and price_below_ema and close[i] < donchian_lower[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_TRAIL_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_TRAIL_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian break below lower band
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Donchian break above upper band
            if close[i] > donchian_upper[i]:
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

# Hypothesis: 6h strategy using Donchian breakout with weekly EMA200 filter and volume confirmation.
# Long when price breaks above 6h Donchian(20) high with weekly EMA200 filter (price > EMA200) and volume > 1.5x average.
# Short when price breaks below 6h Donchian(20) low with weekly EMA200 filter (price < EMA200) and volume > 1.5x average.
# Exit on opposite Donchian break or trailing stop at 2x ATR.
# Weekly EMA200 provides trend filter that works in both bull and bear markets.
# Donchian breakouts capture momentum, volume confirms strength, and weekly filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13887_6h_donchian20_weekly_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 200
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 2.0

def calculate_donchian(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(values, period):
    """Calculate EMA"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load weekly data for EMA200 filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200
    close_weekly = df_weekly['close'].values
    ema200_weekly = calculate_ema(close_weekly, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA200 to 6h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # 6h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # ATR for trailing stop
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema200_weekly_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check trailing stop
        if position == 1:  # long position
            # Update stop price to trail behind
            stop_price = max(stop_price, high[i] - (ATR_TRAIL_MULTIPLIER * atr[i]))
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Update stop price to trail behind
            stop_price = min(stop_price, low[i] + (ATR_TRAIL_MULTIPLIER * atr[i]))
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Weekly EMA200 trend filter
        price_above_ema = close[i] > ema200_weekly_aligned[i]
        price_below_ema = close[i] < ema200_weekly_aligned[i]
        
        # Donchian breakout signals
        long_signal = volume_ok and price_above_ema and close[i] > donchian_upper[i]
        short_signal = volume_ok and price_below_ema and close[i] < donchian_lower[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_TRAIL_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_TRAIL_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian break below lower band
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Donchian break above upper band
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals