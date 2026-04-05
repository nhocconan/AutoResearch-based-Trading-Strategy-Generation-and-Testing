#!/usr/bin/env python3
"""
Experiment #7895: 6-hour Williams Alligator + Elder Ray with weekly pivot regime filter.
Hypothesis: The Williams Alligator identifies trending vs ranging markets (jaws-teeth-lips alignment), Elder Ray measures bull/bear power via EMA13, and weekly pivot provides structural bias. In trending markets (Alligator aligned), we trade Elder Ray extremes; in ranging markets, we fade at weekly pivot S3/R3 levels. This combination adapts to market regimes, reducing whipsaw in sideways markets while capturing trends. Targets 100-200 trades over 4 years with Williams %R for entry timing and ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7895_6h_alligator_elder_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Williams Alligator parameters (standard)
JAWS_PERIOD = 13  # slowest
TEETH_PERIOD = 8  # medium
LIPS_PERIOD = 5   # fastest

# Elder Ray parameters
ELDER_EMA_PERIOD = 13

# Weekly pivot parameters
PIVOT_LOOKBACK = 5  # weeks for pivot calculation

# Risk management
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

def calculate_williams_r(high, low, close, period):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    wr = wr.replace(0, np.nan).fillna(-50)  # neutral when no range
    return wr.values

def calculate_alligator_lines(high, low, close, jaws_p, teeth_p, lips_p):
    """Williams Alligator: SMMA of median price (H+L)/2"""
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaws = pd.Series(median_price).ewm(span=jaws_p, adjust=False, min_periods=jaws_p).mean()
    teeth = pd.Series(median_price).ewm(span=teeth_p, adjust=False, min_periods=teeth_p).mean()
    lips = pd.Series(median_price).ewm(span=lips_p, adjust=False, min_periods=lips_p).mean()
    
    return jaws.values, teeth.values, lips.values

def calculate_elder_ray(close, ema_period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    bull_power = high - ema.values  # will be set after high is available
    bear_power = low - ema.values
    return bull_power, bear_power, ema.values

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P = (H+L+C)/3, S1=2P-H, R1=2P-L, etc."""
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3
    pivot = (typical_price + high + low) / 3  # standard formula: (H+L+C)/3
    
    # Support and resistance levels
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    s3 = low - 2 * (high - low)
    r3 = high + 2 * (high - low)
    
    return pivot, s1, r1, s2, r2, s3, r3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (using weekly OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot, s1, r1, s2, r2, s3, r3 = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaws, teeth, lips = calculate_alligator_lines(high, low, close, JAWS_PERIOD, TEETH_PERIOD, LIPS_PERIOD)
    
    # Elder Ray
    ema_13 = pd.Series(close).ewm(span=ELDER_EMA_PERIOD, adjust=False, min_periods=ELDER_EMA_PERIOD).mean()
    bull_power = high - ema_13.values
    bear_power = low - ema_13.values
    
    # Williams %R for entry timing
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    
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
    start = max(JAWS_PERIOD, TEETH_PERIOD, LIPS_PERIOD, ELDER_EMA_PERIOD, WILLIAMS_R_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
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
        
        # Market regime detection via Williams Alligator
        # Trending: jaws > teeth > lips (downtrend) OR lips > teeth > jaws (uptrend)
        # Ranging: otherwise (intertwined)
        jaws_val = jaws[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check for clear alignment (trending market)
        is_uptrend_aligned = (lips_val > teeth_val) and (teeth_val > jaws_val)
        is_downtrend_aligned = (jaws_val > teeth_val) and (teeth_val > lips_val)
        is_trending = is_uptrend_aligned or is_downtrend_aligned
        is_ranging = not is_trending
        
        # Williams %R conditions for entry timing
        wr_oversold = williams_r[i] <= WILLIAMS_R_OVERSOLD
        wr_overbought = williams_r[i] >= WILLIAMS_R_OVERBOUGHT
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > np.nanpercentile(bull_power[max(0, i-50):i+1], 70) if i >= 50 else bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0 and abs(bear_power[i]) > np.nanpercentile(abs(bear_power[max(0, i-50):i+1]), 70) if i >= 50 else bear_power[i] < 0
        
        # Weekly pivot fade/breakout conditions
        near_s3 = close[i] <= s3_aligned[i] * 1.002  # within 0.2% of S3
        near_r3 = close[i] >= r3_aligned[i] * 0.998  # within 0.2% of R3
        break_s3 = close[i] < s3_aligned[i]  # breaking below S3
        break_r3 = close[i] > r3_aligned[i]  # breaking above R3
        
        # Entry logic based on market regime
        if is_trending:
            # In trending markets: trade Elder Ray extremes with Williams %R timing
            long_entry = is_uptrend_aligned and strong_bull_power and wr_oversold
            short_entry = is_downtrend_aligned and strong_bear_power and wr_overbought
        else:
            # In ranging markets: fade at weekly S3/R3 levels
            long_entry = is_ranging and near_s3 and wr_oversold
            short_entry = is_ranging and near_r3 and wr_overbought
        
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
Experiment #7895: 6-hour Williams Alligator + Elder Ray with weekly pivot regime filter.
Hypothesis: The Williams Alligator identifies trending vs ranging markets (jaws-teeth-lips alignment), Elder Ray measures bull/bear power via EMA13, and weekly pivot provides structural bias. In trending markets (Alligator aligned), we trade Elder Ray extremes; in ranging markets, we fade at weekly pivot S3/R3 levels. This combination adapts to market regimes, reducing whipsaw in sideways markets while capturing trends. Targets 100-200 trades over 4 years with Williams %R for entry timing and ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7895_6h_alligator_elder_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Williams Alligator parameters (standard)
JAWS_PERIOD = 13  # slowest
TEETH_PERIOD = 8  # medium
LIPS_PERIOD = 5   # fastest

# Elder Ray parameters
ELDER_EMA_PERIOD = 13

# Weekly pivot parameters
PIVOT_LOOKBACK = 5  # weeks for pivot calculation

# Risk management
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

def calculate_williams_r(high, low, close, period):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    wr = wr.replace(0, np.nan).fillna(-50)  # neutral when no range
    return wr.values

def calculate_alligator_lines(high, low, close, jaws_p, teeth_p, lips_p):
    """Williams Alligator: SMMA of median price (H+L)/2"""
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaws = pd.Series(median_price).ewm(span=jaws_p, adjust=False, min_periods=jaws_p).mean()
    teeth = pd.Series(median_price).ewm(span=teeth_p, adjust=False, min_periods=teeth_p).mean()
    lips = pd.Series(median_price).ewm(span=lips_p, adjust=False, min_periods=lips_p).mean()
    
    return jaws.values, teeth.values, lips.values

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P = (H+L+C)/3, S1=2P-H, R1=2P-L, etc."""
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3
    pivot = (typical_price + high + low) / 3  # standard formula: (H+L+C)/3
    
    # Support and resistance levels
    s1 = 2 * pivot - high
    r1 = 2 * pivot - low
    s2 = pivot - (high - low)
    r2 = pivot + (high - low)
    s3 = low - 2 * (high - low)
    r3 = high + 2 * (high - low)
    
    return pivot, s1, r1, s2, r2, s3, r3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (using weekly OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot, s1, r1, s2, r2, s3, r3 = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaws, teeth, lips = calculate_alligator_lines(high, low, close, JAWS_PERIOD, TEETH_PERIOD, LIPS_PERIOD)
    
    # Elder Ray
    ema_13 = pd.Series(close).ewm(span=ELDER_EMA_PERIOD, adjust=False, min_periods=ELDER_EMA_PERIOD).mean()
    bull_power = high - ema_13.values
    bear_power = low - ema_13.values
    
    # Williams %R for entry timing
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    
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
    start = max(JAWS_PERIOD, TEETH_PERIOD, LIPS_PERIOD, ELDER_EMA_PERIOD, WILLIAMS_R_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
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
        
        # Market regime detection via Williams Alligator
        # Trending: jaws > teeth > lips (downtrend) OR lips > teeth > jaws (uptrend)
        # Ranging: otherwise (intertwined)
        jaws_val = jaws[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Check for clear alignment (trending market)
        is_uptrend_aligned = (lips_val > teeth_val) and (teeth_val > jaws_val)
        is_downtrend_aligned = (jaws_val > teeth_val) and (teeth_val > lips_val)
        is_trending = is_uptrend_aligned or is_downtrend_aligned
        is_ranging = not is_trending
        
        # Williams %R conditions for entry timing
        wr_oversold = williams_r[i] <= WILLIAMS_R_OVERSOLD
        wr_overbought = williams_r[i] >= WILLIAMS_R_OVERBOUGHT
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > np.nanpercentile(bull_power[max(0, i-50):i+1], 70) if i >= 50 else bull_power[i] > 0
        strong_bear_power = bear_power[i] < 0 and abs(bear_power[i]) > np.nanpercentile(abs(bear_power[max(0, i-50):i+1]), 70) if i >= 50 else bear_power[i] < 0
        
        # Weekly pivot fade/breakout conditions
        near_s3 = close[i] <= s3_aligned[i] * 1.002  # within 0.2% of S3
        near_r3 = close[i] >= r3_aligned[i] * 0.998  # within 0.2% of R3
        break_s3 = close[i] < s3_aligned[i]  # breaking below S3
        break_r3 = close[i] > r3_aligned[i]  # breaking above R3
        
        # Entry logic based on market regime
        if is_trending:
            # In trending markets: trade Elder Ray extremes with Williams %R timing
            long_entry = is_uptrend_aligned and strong_bull_power and wr_oversold
            short_entry = is_downtrend_aligned and strong_bear_power and wr_overbought
        else:
            # In ranging markets: fade at weekly S3/R3 levels
            long_entry = is_ranging and near_s3 and wr_oversold
            short_entry = is_ranging and near_r3 and wr_overbought
        
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