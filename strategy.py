#!/usr/bin/env python3
"""
Experiment #7954: 1-hour momentum with 4h/1d trend and volume confirmation.
Hypothesis: In 1h timeframe, price momentum (close > open) combined with 4h trend filter (price above/below 4h EMA50) and 1d volume confirmation (volume > 1.5x 20-period MA) captures trending moves while minimizing false signals. The 4h/1d timeframes provide directional bias, and 1h is used only for entry timing. Volume confirmation reduces noise. Target: 60-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7954_1h_momentum_4h_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).ewm(span=VOLUME_MA_PERIOD, adjust=False, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h EMA
        bull_bias = ema_4h_aligned[i] == 1   # 4h close above EMA50
        bear_bias = ema_4h_aligned[i] == -1  # 4h close below EMA50
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Momentum condition: close > open (bullish candle) or close < open (bearish candle)
        bullish_candle = close[i] > prices['open'].iloc[i]
        bearish_candle = close[i] < prices['open'].iloc[i]
        
        # Entry conditions
        long_entry = bull_bias and volume_confirmed and bullish_candle
        short_entry = bear_bias and volume_confirmed and bearish_candle
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
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
Experiment #7954: 1-hour momentum with 4h/1d trend and volume confirmation.
Hypothesis: In 1h timeframe, price momentum (close > open) combined with 4h trend filter (price above/below 4h EMA50) and 1d volume confirmation (volume > 1.5x 20-period MA) captures trending moves while minimizing false signals. The 4h/1d timeframes provide directional bias, and 1h is used only for entry timing. Volume confirmation reduces noise. Target: 60-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7954_1h_momentum_4h_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_vs_ema_4h = np.where(close_4h > ema_4h, 1, -1)  # 1=bullish, -1=bearish
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, price_vs_ema_4h)
    
    # Calculate 1d volume moving average
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).ewm(span=VOLUME_MA_PERIOD, adjust=False, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 4h EMA
        bull_bias = ema_4h_aligned[i] == 1   # 4h close above EMA50
        bear_bias = ema_4h_aligned[i] == -1  # 4h close below EMA50
        
        # Volume confirmation from 1d
        volume_confirmed = volume[i] > (volume_ma_1d_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma_1d_aligned[i]) else False
        
        # Momentum condition: close > open (bullish candle) or close < open (bearish candle)
        bullish_candle = close[i] > prices['open'].iloc[i]
        bearish_candle = close[i] < prices['open'].iloc[i]
        
        # Entry conditions
        long_entry = bull_bias and volume_confirmed and bullish_candle
        short_entry = bear_bias and volume_confirmed and bearish_candle
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>