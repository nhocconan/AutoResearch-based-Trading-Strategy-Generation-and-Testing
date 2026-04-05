#!/usr/bin/env python3
"""
Experiment #8274: 1-hour momentum with 4h/1d trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price momentum on 1h aligned with higher timeframe 
trend (4h EMA50 for direction, 1d ADX for strength) and confirmed by volume spikes captures 
sustained moves while avoiding whipsaw. The 4h EMA provides trend direction, 1d ADX > 25 
filters for trending markets, and volume > 1.5x 20MA confirms institutional interest. 
Targeting 60-150 total trades over 4 years (15-37/year) for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8274_1h_momentum_4h_1d_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_PERIOD_4H = 50
ADX_PERIOD_1D = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD_4H, adjust=False, min_periods=EMA_PERIOD_4H).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    up_smoothed = pd.Series(up_move).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    down_smoothed = pd.Series(down_move).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Indicators
    plus_di = 100 * up_smoothed / atr_1d
    minus_di = 100 * down_smoothed / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price momentum (1h ROC)
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100
    
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
    start = max(EMA_PERIOD_4H, ADX_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trend conditions
        price_above_ema = close[i] > ema_4h_aligned[i]
        price_below_ema = close[i] < ema_4h_aligned[i]
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Momentum confirmation (positive for long, negative for short)
        momentum_long = roc[i] > 0
        momentum_short = roc[i] < 0
        
        # Entry conditions
        long_entry = price_above_ema and strong_trend and volume_confirmed and momentum_long
        short_entry = price_below_ema and strong_trend and volume_confirmed and momentum_short
        
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
Experiment #8274: 1-hour momentum with 4h/1d trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price momentum on 1h aligned with higher timeframe 
trend (4h EMA50 for direction, 1d ADX for strength) and confirmed by volume spikes captures 
sustained moves while avoiding whipsaw. The 4h EMA provides trend direction, 1d ADX > 25 
filters for trending markets, and volume > 1.5x 20MA confirms institutional interest. 
Targeting 60-150 total trades over 4 years (15-37/year) for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8274_1h_momentum_4h_1d_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_PERIOD_4H = 50
ADX_PERIOD_1D = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD_4H, adjust=False, min_periods=EMA_PERIOD_4H).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    up_smoothed = pd.Series(up_move).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    down_smoothed = pd.Series(down_move).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Indicators
    plus_di = 100 * up_smoothed / atr_1d
    minus_di = 100 * down_smoothed / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price momentum (1h ROC)
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100
    
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
    start = max(EMA_PERIOD_4H, ADX_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trend conditions
        price_above_ema = close[i] > ema_4h_aligned[i]
        price_below_ema = close[i] < ema_4h_aligned[i]
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Momentum confirmation (positive for long, negative for short)
        momentum_long = roc[i] > 0
        momentum_short = roc[i] < 0
        
        # Entry conditions
        long_entry = price_above_ema and strong_trend and volume_confirmed and momentum_long
        short_entry = price_below_ema and strong_trend and volume_confirmed and momentum_short
        
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