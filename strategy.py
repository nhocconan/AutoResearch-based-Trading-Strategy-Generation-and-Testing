# 1d strategy using 1-week trend filter with volume confirmation and ATR-based risk management
# Uses weekly EMA trend direction, daily price action for entry, and volume confirmation
# Designed for 30-100 trades over 4 years to minimize fee drag while capturing major trends
# Works in both bull and bear markets by following weekly trend and using volatility filters

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13884_1d_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_HOLD_DAYS = 3  # Minimum holding period to reduce churn

def calculate_ema(close, period):
    """Calculate EMA with proper smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Daily data for entry signals and risk management
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Check stops (only after minimum hold period)
        if position == 1:  # long position
            if bars_since_entry >= MIN_HOLD_DAYS:
                # Check stop loss
                if close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    continue
        
        elif position == -1:  # short position
            if bars_since_entry >= MIN_HOLD_DAYS:
                # Check stop loss
                if close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: weekly EMA direction
        # Use slope of weekly EMA to determine trend
        if i >= 2:
            weekly_ema_slope = weekly_ema_aligned[i] - weekly_ema_aligned[i-1]
            trend_up = weekly_ema_slope > 0
            trend_down = weekly_ema_slope < 0
        else:
            trend_up = weekly_ema_aligned[i] > weekly_ema_aligned[i-1] if i > 0 else False
            trend_down = weekly_ema_aligned[i] < weekly_ema_aligned[i-1] if i > 0 else False
        
        # Entry signals: price rejection from weekly EMA with volume
        # Long: price bounces above weekly EMA in uptrend with volume
        # Short: price rejects below weekly EMA in downtrend with volume
        price_above_weekly_ema = close[i] > weekly_ema_aligned[i]
        price_below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        long_signal = volume_ok and trend_up and price_above_weekly_ema
        short_signal = volume_ok and trend_down and price_below_weekly_ema
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below weekly EMA
            if close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above weekly EMA
            if close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13884_1d_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MIN_HOLD_DAYS = 3  # Minimum holding period to reduce churn

def calculate_ema(close, period):
    """Calculate EMA with proper smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

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
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Daily data for entry signals and risk management
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        # Check stops (only after minimum hold period)
        if position == 1:  # long position
            if bars_since_entry >= MIN_HOLD_DAYS:
                # Check stop loss
                if close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    continue
        
        elif position == -1:  # short position
            if bars_since_entry >= MIN_HOLD_DAYS:
                # Check stop loss
                if close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend filter: weekly EMA direction
        # Use slope of weekly EMA to determine trend
        if i >= 2:
            weekly_ema_slope = weekly_ema_aligned[i] - weekly_ema_aligned[i-1]
            trend_up = weekly_ema_slope > 0
            trend_down = weekly_ema_slope < 0
        else:
            trend_up = weekly_ema_aligned[i] > weekly_ema_aligned[i-1] if i > 0 else False
            trend_down = weekly_ema_aligned[i] < weekly_ema_aligned[i-1] if i > 0 else False
        
        # Entry signals: price rejection from weekly EMA with volume
        # Long: price bounces above weekly EMA in uptrend with volume
        # Short: price rejects below weekly EMA in downtrend with volume
        price_above_weekly_ema = close[i] > weekly_ema_aligned[i]
        price_below_weekly_ema = close[i] < weekly_ema_aligned[i]
        
        long_signal = volume_ok and trend_up and price_above_weekly_ema
        short_signal = volume_ok and trend_down and price_below_weekly_ema
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on close below weekly EMA
            if close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above weekly EMA
            if close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals