#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points for direction and 6h Donchian(20) breakouts with volume confirmation.
# Uses weekly pivot levels (R4/S4) for breakouts and (R3/S3) for mean reversion, filtered by weekly trend.
# Designed for ~100-150 total trades over 4 years (25-38/year) to avoid excessive fees.
# Works in bull (breakouts above R4 with volume) and bear (breakdowns below S4 with volume) markets.
# Target: 100-200 total trades, 0.25 position size, max DD < -50%.

name = "exp_13735_6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # Days to look back for weekly pivot calculation
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points from daily OHLC"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = high + 3 * (pivot - low)
    s4 = low - 3 * (high - pivot)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily OHLC
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate weekly pivot points (using weekly aggregation of daily data)
    # We'll compute pivot points for each week using the last 5 days
    pivot_points = np.full_like(close_daily, np.nan)
    r3_levels = np.full_like(close_daily, np.nan)
    s3_levels = np.full_like(close_daily, np.nan)
    r4_levels = np.full_like(close_daily, np.nan)
    s4_levels = np.full_like(close_daily, np.nan)
    
    for i in range(len(close_daily)):
        if i >= PIVOT_LOOKBACK:
            # Use last 5 days to calculate weekly pivot
            lookback_high = np.max(high_daily[i-PIVOT_LOOKBACK+1:i+1])
            lookback_low = np.min(low_daily[i-PIVOT_LOOKBACK+1:i+1])
            lookback_close = close_daily[i]
            pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(lookback_high, lookback_low, lookback_close)
            pivot_points[i] = pivot
            r3_levels[i] = r3
            s3_levels[i] = s3
            r4_levels[i] = r4
            s4_levels[i] = s4
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot_points)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_levels)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_levels)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_levels)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_levels)
    
    # Weekly trend (EMA50 on weekly closes)
    close_weekly = df_weekly['close'].values
    weekly_ema = calculate_ema(close_weekly, 50)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 6h Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 6h
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 50, VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
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
        
        # Trend direction from weekly EMA
        above_weekly_ema = close > weekly_ema_aligned[i]
        below_weekly_ema = close < weekly_ema_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long breakout: price breaks above R4 with volume in uptrend
            long_breakout = volume_ok and above_weekly_ema and close > r4_aligned[i] and close <= r4_aligned[i-1]
            
            # Short breakdown: price breaks below S4 with volume in downtrend
            short_breakdown = volume_ok and below_weekly_ema and close < s4_aligned[i] and close >= s4_aligned[i-1]
            
            # Mean reversion longs: price touches S3 with volume in uptrend
            long_mean_reversion = volume_ok and above_weekly_ema and close <= s3_aligned[i] and close > s3_aligned[i-1]
            
            # Mean reversion shorts: price touches R3 with volume in downtrend
            short_mean_reversion = volume_ok and below_weekly_ema and close >= r3_aligned[i] and close < r3_aligned[i-1]
        else:
            long_breakout = False
            short_breakdown = False
            long_mean_reversion = False
            short_mean_reversion = False
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakdown:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_mean_reversion:
                signals[i] = SIGNAL_SIZE * 0.5  # Half position for mean reversion
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_mean_reversion:
                signals[i] = -SIGNAL_SIZE * 0.5  # Half position for mean reversion
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite signals
            exit_signal = (close < s3_aligned[i] and close > s3_aligned[i-1]) or \
                         (close > r4_aligned[i] and close <= r4_aligned[i-1]) or \
                         (close < donchian_low[i-1] and close >= donchian_low[i-2] if i > 1 else False)
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signals
            exit_signal = (close > r3_aligned[i] and close < r3_aligned[i-1]) or \
                         (close < s4_aligned[i] and close >= s4_aligned[i-1]) or \
                         (close > donchian_high[i-1] and close <= donchian_high[i-2] if i > 1 else False)
            if exit_signal:
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

# Hypothesis: 6h strategy using weekly pivot points for direction and 6h Donchian(20) breakouts with volume confirmation.
# Uses weekly pivot levels (R4/S4) for breakouts and (R3/S3) for mean reversion, filtered by weekly trend.
# Designed for ~100-150 total trades over 4 years (25-38/year) to avoid excessive fees.
# Works in bull (breakouts above R4 with volume) and bear (breakdowns below S4 with volume) markets.
# Target: 100-200 total trades, 0.25 position size, max DD < -50%.

name = "exp_13735_6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # Days to look back for weekly pivot calculation
VOLUME_MA_PERIOD = 10
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points from daily OHLC"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = high + 3 * (pivot - low)
    s4 = low - 3 * (high - pivot)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily OHLC
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate weekly pivot points (using weekly aggregation of daily data)
    # We'll compute pivot points for each week using the last 5 days
    pivot_points = np.full_like(close_daily, np.nan)
    r3_levels = np.full_like(close_daily, np.nan)
    s3_levels = np.full_like(close_daily, np.nan)
    r4_levels = np.full_like(close_daily, np.nan)
    s4_levels = np.full_like(close_daily, np.nan)
    
    for i in range(len(close_daily)):
        if i >= PIVOT_LOOKBACK:
            # Use last 5 days to calculate weekly pivot
            lookback_high = np.max(high_daily[i-PIVOT_LOOKBACK+1:i+1])
            lookback_low = np.min(low_daily[i-PIVOT_LOOKBACK+1:i+1])
            lookback_close = close_daily[i]
            pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivot_points(lookback_high, lookback_low, lookback_close)
            pivot_points[i] = pivot
            r3_levels[i] = r3
            s3_levels[i] = s3
            r4_levels[i] = r4
            s4_levels[i] = s4
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot_points)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3_levels)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3_levels)
    r4_aligned = align_htf_to_ltf(prices, df_daily, r4_levels)
    s4_aligned = align_htf_to_ltf(prices, df_daily, s4_levels)
    
    # Weekly trend (EMA50 on weekly closes)
    close_weekly = df_weekly['close'].values
    weekly_ema = calculate_ema(close_weekly, 50)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # 6h Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for 6h
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 50, VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
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
        
        # Trend direction from weekly EMA
        above_weekly_ema = close > weekly_ema_aligned[i]
        below_weekly_ema = close < weekly_ema_aligned[i]
        
        # Donchian breakout signals
        if i > 0 and not np.isnan(donchian_high[i-1]) and not np.isnan(donchian_low[i-1]):
            high_prev = donchian_high[i-1]
            low_prev = donchian_low[i-1]
            
            # Long breakout: price breaks above R4 with volume in uptrend
            long_breakout = volume_ok and above_weekly_ema and close > r4_aligned[i] and close <= r4_aligned[i-1]
            
            # Short breakdown: price breaks below S4 with volume in downtrend
            short_breakdown = volume_ok and below_weekly_ema and close < s4_aligned[i] and close >= s4_aligned[i-1]
            
            # Mean reversion longs: price touches S3 with volume in uptrend
            long_mean_reversion = volume_ok and above_weekly_ema and close <= s3_aligned[i] and close > s3_aligned[i-1]
            
            # Mean reversion shorts: price touches R3 with volume in downtrend
            short_mean_reversion = volume_ok and below_weekly_ema and close >= r3_aligned[i] and close < r3_aligned[i-1]
        else:
            long_breakout = False
            short_breakdown = False
            long_mean_reversion = False
            short_mean_reversion = False
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakdown:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif long_mean_reversion:
                signals[i] = SIGNAL_SIZE * 0.5  # Half position for mean reversion
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_mean_reversion:
                signals[i] = -SIGNAL_SIZE * 0.5  # Half position for mean reversion
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on opposite signals
            exit_signal = (close < s3_aligned[i] and close > s3_aligned[i-1]) or \
                         (close > r4_aligned[i] and close <= r4_aligned[i-1]) or \
                         (close < donchian_low[i-1] and close >= donchian_low[i-2] if i > 1 else False)
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on opposite signals
            exit_signal = (close > r3_aligned[i] and close < r3_aligned[i-1]) or \
                         (close < s4_aligned[i] and close >= s4_aligned[i-1]) or \
                         (close > donchian_high[i-1] and close <= donchian_high[i-2] if i > 1 else False)
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>