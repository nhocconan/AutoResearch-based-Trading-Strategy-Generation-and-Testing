#!/usr/bin/env python3
"""
Experiment #10935: 6h Camarilla Pivot Reversal with Weekly Trend and Volume Filter
Hypothesis: Camarilla pivot levels (R3/S3) act as strong reversal zones in ranging markets,
while weekly trend filter prevents counter-trend trades during strong trends. Volume confirmation
ensures institutional participation. Designed for 6H timeframe to target 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10935_6h_camarilla_pivot_reversal_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1   # Use previous day's OHLC for pivot calculation
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels from previous period's OHLC"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_ = high - low
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 2)  # R3 = pivot + (range * 1.1/2)
    s3 = pivot - (range_ * 1.1 / 2)  # S3 = pivot - (range * 1.1/2)
    r4 = pivot + (range_ * 1.1)      # R4 = pivot + (range * 1.1)
    s4 = pivot - (range_ * 1.1)      # S4 = pivot - (range * 1.1)
    return r3, s3, r4, s4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend
    ema_weekly = calculate_ema(df_weekly['close'].values, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Shift OHLC by 1 to get previous period's values for Camarilla calculation
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN as there's no previous period
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r3, camarilla_s3, camarilla_r4, camarilla_s4 = calculate_camarilla_pivots(high_prev, low_prev, close_prev)
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # Camarilla reversal conditions
        # Long when price touches or goes below S3 and shows rejection (close > open)
        touch_s3 = low[i] <= camarilla_s3[i] if not np.isnan(camarilla_s3[i]) else False
        reject_s3 = close[i] > open[i] if not np.isnan(open[i]) else False  # Bullish candle
        # Short when price touches or goes above R3 and shows rejection (close < open)
        touch_r3 = high[i] >= camarilla_r3[i] if not np.isnan(camarilla_r3[i]) else False
        reject_r3 = close[i] < open[i] if not np.isnan(open[i]) else False  # Bearish candle
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (weekly) - only trade in direction of weekly trend
        uptrend_weekly = close[i] > ema_weekly_aligned[i]
        downtrend_weekly = close[i] < ema_weekly_aligned[i]
        
        # Entry conditions - reversal trades with trend filter
        long_entry = touch_s3 and reject_s3 and volume_ok and downtrend_weekly  # Buy weakness in downtrend
        short_entry = touch_r3 and reject_r3 and volume_ok and uptrend_weekly   # Sell strength in uptrend
        
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
Experiment #10935: 6h Donchian Breakout with Weekly Trend and Volume Filter
Hypothesis: Donchian(20) breakouts capture strong directional moves. Weekly EMA provides trend bias
to avoid counter-trend trades, and volume filter ensures institutional participation. Designed for
6H timeframe to target 50-150 trades over 4 years (12-37/year). Works in bull (breakouts continue)
and bear (breakouts reverse quickly) by using weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10935_6h_donchian20_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend
    ema_weekly = calculate_ema(df_weekly['close'].values, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]):
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
        
        # Donchian breakout conditions
        breakout_up = high[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_down = low[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter (weekly)
        uptrend_weekly = close[i] > ema_weekly_aligned[i]
        downtrend_weekly = close[i] < ema_weekly_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_ok and uptrend_weekly
        short_entry = breakout_down and volume_ok and downtrend_weekly
        
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

{
  "name": "exp_10935_6h_donchian20_1w_ema_vol_v1",
  "timeframe": "6h",
  "leverage": 1.0,
  "donchian_period": 20,
  "weekly_ema_period": 21,
  "volume_ma_period": 20,
  "volume_threshold": 1.5,
  "signal_size": 0.25,
  "atr_period": 14,
  "atr_stop_multiplier": 2.0
}