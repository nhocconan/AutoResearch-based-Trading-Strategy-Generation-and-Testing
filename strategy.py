#!/usr/bin/env python3
"""
Experiment #8239: 6-hour Williams %R with 12-hour Trend Filter and Volume Confirmation.
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
In trending markets (identified by 12h price above/below 20-period EMA), extreme 
Williams %R readings (>80 for oversold, <20 for overbought) with volume confirmation 
provide high-probability mean reversion entries. Works in both bull and bear markets 
by fading extremes in the direction of the higher timeframe trend, reducing whipsaw.
Targets 75-150 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8239_6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
TREND_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=WILLIAMS_PERIOD, min_periods=WILLIAMS_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=WILLIAMS_PERIOD, min_periods=WILLIAMS_PERIOD).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
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
    start = max(WILLIAMS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, TREND_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h close above EMA20
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h close below EMA20
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Williams %R conditions - extreme readings
        oversold = williams_r[i] <= WILLIAMS_OVERSOLD   # <= -80
        overbought = williams_r[i] >= WILLIAMS_OVERBOUGHT  # >= -20
        
        # Entry conditions: fade extremes in direction of trend
        long_entry = bull_bias and oversold and volume_confirmed
        short_entry = bear_bias and overbought and volume_confirmed
        
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
Experiment #8239: 6-hour Williams %R with 12-hour Trend Filter and Volume Confirmation.
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
In trending markets (identified by 12h price above/below 20-period EMA), extreme 
Williams %R readings (>80 for oversold, <20 for overbought) with volume confirmation 
provide high-probability mean reversion entries. Works in both bull and bear markets 
by fading extremes in the direction of the higher timeframe trend, reducing whipsaw.
Targets 75-150 total trades over 4 years for optimal balance of signal quality and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8239_6h_williamsr_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
TREND_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=WILLIAMS_PERIOD, min_periods=WILLIAMS_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=WILLIAMS_PERIOD, min_periods=WILLIAMS_PERIOD).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
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
    start = max(WILLIAMS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, TREND_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h close above EMA20
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h close below EMA20
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Williams %R conditions - extreme readings
        oversold = williams_r[i] <= WILLIAMS_OVERSOLD   # <= -80
        overbought = williams_r[i] >= WILLIAMS_OVERBOUGHT  # >= -20
        
        # Entry conditions: fade extremes in direction of trend
        long_entry = bull_bias and oversold and volume_confirmed
        short_entry = bear_bias and overbought and volume_confirmed
        
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