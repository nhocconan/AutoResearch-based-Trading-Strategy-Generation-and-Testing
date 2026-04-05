# 2025-06-08
# 7558
# 1d
# 1w
#!/usr/bin/env python3
"""
exp_7558_1d_1w_ema200_trend_follow_v1
Hypothesis: On the daily timeframe, use weekly EMA200 as the primary trend filter.
In bull markets (price > weekly EMA200): go long on daily close above daily EMA50.
In bear markets (price < weekly EMA200): go short on daily close below daily EMA50.
Requires volume confirmation (volume > 1.5x 20-day average) to avoid false breakouts.
Uses ATR-based stoploss (2x ATR) and scales out at 2x ATR profit.
Targets 30-100 trades over 4 years (7-25/year) with strict trend + momentum conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7558_1d_1w_ema200_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_TREND = 200
DAILY_EMA_FAST = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_SCALE_OUT_MULTIPLIER = 2.0  # scale out half position at 2x ATR profit

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    weekly_ema_200 = pd.Series(close_1w).ewm(span=WEEKLY_EMA_TREND, adjust=False, min_periods=WEEKLY_EMA_TREND).mean().values
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_200)
    
    # Calculate LTF (daily) indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for entry signal
    daily_ema_50 = pd.Series(close).ewm(span=DAILY_EMA_FAST, adjust=False, min_periods=DAILY_EMA_FAST).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_TREND, DAILY_EMA_FAST, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_ema_200_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check stoploss or scale out
        if position == 1:  # long position
            if close[i] <= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] >= entry_price + (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] <= entry_price - (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = -SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        
        # Determine market regime from weekly trend
        bull_regime = close[i] > weekly_ema_200_aligned[i]   # price above weekly EMA200
        bear_regime = close[i] < weekly_ema_200_aligned[i]   # price below weekly EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry signals: EMA50 cross with volume confirmation
        ema_cross_up = (close[i] > daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] <= daily_ema_50[i-1])
        ema_cross_down = (close[i] < daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] >= daily_ema_50[i-1])
        
        # Entry conditions
        long_entry = bull_regime and ema_cross_up and volume_confirmed
        short_entry = bear_regime and ema_cross_down and volume_confirmed
        
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

# 2025-06-08
# 7558
# 1d
# 1w
#!/usr/bin/env python3
"""
exp_7558_1d_1w_ema200_trend_follow_v1
Hypothesis: On the daily timeframe, use weekly EMA200 as the primary trend filter.
In bull markets (price > weekly EMA200): go long on daily close above daily EMA50.
In bear markets (price < weekly EMA200): go short on daily close below daily EMA50.
Requires volume confirmation (volume > 1.5x 20-day average) to avoid false breakouts.
Uses ATR-based stoploss (2x ATR) and scales out at 2x ATR profit.
Targets 30-100 trades over 4 years (7-25/year) with strict trend + momentum conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7558_1d_1w_ema200_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_TREND = 200
DAILY_EMA_FAST = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_SCALE_OUT_MULTIPLIER = 2.0  # scale out half position at 2x ATR profit

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    weekly_ema_200 = pd.Series(close_1w).ewm(span=WEEKLY_EMA_TREND, adjust=False, min_periods=WEEKLY_EMA_TREND).mean().values
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_200)
    
    # Calculate LTF (daily) indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for entry signal
    daily_ema_50 = pd.Series(close).ewm(span=DAILY_EMA_FAST, adjust=False, min_periods=DAILY_EMA_FAST).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_TREND, DAILY_EMA_FAST, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_ema_200_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check stoploss or scale out
        if position == 1:  # long position
            if close[i] <= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] >= entry_price + (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] <= entry_price - (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = -SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        
        # Determine market regime from weekly trend
        bull_regime = close[i] > weekly_ema_200_aligned[i]   # price above weekly EMA200
        bear_regime = close[i] < weekly_ema_200_aligned[i]   # price below weekly EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry signals: EMA50 cross with volume confirmation
        ema_cross_up = (close[i] > daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] <= daily_ema_50[i-1])
        ema_cross_down = (close[i] < daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] >= daily_ema_50[i-1])
        
        # Entry conditions
        long_entry = bull_regime and ema_cross_up and volume_confirmed
        short_entry = bear_regime and ema_cross_down and volume_confirmed
        
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

# 2025-06-08
# 7558
# 1d
# 1w
#!/usr/bin/env python3
"""
exp_7558_1d_1w_ema200_trend_follow_v1
Hypothesis: On the daily timeframe, use weekly EMA200 as the primary trend filter.
In bull markets (price > weekly EMA200): go long on daily close above daily EMA50.
In bear markets (price < weekly EMA200): go short on daily close below daily EMA50.
Requires volume confirmation (volume > 1.5x 20-day average) to avoid false breakouts.
Uses ATR-based stoploss (2x ATR) and scales out at 2x ATR profit.
Targets 30-100 trades over 4 years (7-25/year) with strict trend + momentum conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7558_1d_1w_ema200_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_TREND = 200
DAILY_EMA_FAST = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_SCALE_OUT_MULTIPLIER = 2.0  # scale out half position at 2x ATR profit

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    weekly_ema_200 = pd.Series(close_1w).ewm(span=WEEKLY_EMA_TREND, adjust=False, min_periods=WEEKLY_EMA_TREND).mean().values
    weekly_ema_200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_200)
    
    # Calculate LTF (daily) indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for entry signal
    daily_ema_50 = pd.Series(close).ewm(span=DAILY_EMA_FAST, adjust=False, min_periods=DAILY_EMA_FAST).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_TREND, DAILY_EMA_FAST, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_ema_200_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Check stoploss or scale out
        if position == 1:  # long position
            if close[i] <= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] >= entry_price + (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:  # stoploss hit
                signals[i] = 0.0
                position = 0
                continue
            elif close[i] <= entry_price - (ATR_SCALE_OUT_MULTIPLIER * atr[i]):  # scale out half
                signals[i] = -SIGNAL_SIZE / 2
                # Keep half position, move stop to breakeven
                stop_price = entry_price
                continue
        
        # Determine market regime from weekly trend
        bull_regime = close[i] > weekly_ema_200_aligned[i]   # price above weekly EMA200
        bear_regime = close[i] < weekly_ema_200_aligned[i]   # price below weekly EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry signals: EMA50 cross with volume confirmation
        ema_cross_up = (close[i] > daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] <= daily_ema_50[i-1])
        ema_cross_down = (close[i] < daily_ema_50[i]) and (i-1 >= 0) and (close[i-1] >= daily_ema_50[i-1])
        
        # Entry conditions
        long_entry = bull_regime and ema_cross_up and volume_confirmed
        short_entry = bear_regime and ema_cross_down and volume_confirmed
        
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