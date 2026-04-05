#!/usr/bin/env python3
"""
Experiment #7714: 1-hour momentum with 4h/1d trend filters and volume confirmation.
Hypothesis: In both bull and bear markets, 1h momentum aligned with higher timeframe trend
(4h EMA and 1d ADX) and confirmed by volume spikes captures meaningful moves while avoiding whipsaw.
Trades are filtered to active session (08-20 UTC) to reduce noise. Targets 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7714_1h_momentum_4h_1d_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
EMA_TREND_4H = 20
ADX_PERIOD_1D = 14
ADX_THRESHOLD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_4H, adjust=False, min_periods=EMA_TREND_4H).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum (rate of change)
    momentum = np.zeros_like(close)
    momentum[MOMENTUM_PERIOD:] = (close[MOMENTUM_PERIOD:] - close[:-MOMENTUM_PERIOD]) / close[:-MOMENTUM_PERIOD] * 100
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, EMA_TREND_4H, ADX_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market conditions
        # Trend direction from 4h EMA
        bull_trend = close[i] > ema_4h_aligned[i]
        bear_trend = close[i] < ema_4h_aligned[i]
        
        # Trend strength from 1d ADX
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Momentum signal
        momentum_signal = momentum[i] if not np.isnan(momentum[i]) else 0
        momentum_up = momentum_signal > 0
        momentum_down = momentum_signal < 0
        
        # Entry conditions
        long_entry = bull_trend and strong_trend and volume_confirmed and momentum_up
        short_entry = bear_trend and strong_trend and volume_confirmed and momentum_down
        
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
Experiment #7714: 1-hour momentum with 4h/1d trend filters and volume confirmation.
Hypothesis: In both bull and bear markets, 1h momentum aligned with higher timeframe trend
(4h EMA and 1d ADX) and confirmed by volume spikes captures meaningful moves while avoiding whipsaw.
Trades are filtered to active session (08-20 UTC) to reduce noise. Targets 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7714_1h_momentum_4h_1d_trend_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
MOMENTUM_PERIOD = 10
EMA_TREND_4H = 20
ADX_PERIOD_1D = 14
ADX_THRESHOLD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_TREND_4H, adjust=False, min_periods=EMA_TREND_4H).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD_1D, adjust=False, min_periods=ADX_PERIOD_1D).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum (rate of change)
    momentum = np.zeros_like(close)
    momentum[MOMENTUM_PERIOD:] = (close[MOMENTUM_PERIOD:] - close[:-MOMENTUM_PERIOD]) / close[:-MOMENTUM_PERIOD] * 100
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(MOMENTUM_PERIOD, EMA_TREND_4H, ADX_PERIOD_1D, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                # Check stop loss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market conditions
        # Trend direction from 4h EMA
        bull_trend = close[i] > ema_4h_aligned[i]
        bear_trend = close[i] < ema_4h_aligned[i]
        
        # Trend strength from 1d ADX
        strong_trend = adx_aligned[i] > ADX_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Momentum signal
        momentum_signal = momentum[i] if not np.isnan(momentum[i]) else 0
        momentum_up = momentum_signal > 0
        momentum_down = momentum_signal < 0
        
        # Entry conditions
        long_entry = bull_trend and strong_trend and volume_confirmed and momentum_up
        short_entry = bear_trend and strong_trend and volume_confirmed and momentum_down
        
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