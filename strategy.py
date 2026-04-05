#!/usr/bin/env python3
"""
exp_7339_6h_adx_regime_donchian_v1
Hypothesis: 6h Donchian(20) breakouts filtered by 12h ADX regime and volume confirmation.
In trending markets (ADX > 25), trade breakouts in direction of trend.
In ranging markets (ADX < 20), fade at Donchian extremes with volume spike.
Uses 12h HTF for regime detection to avoid whipsaws. Designed for low trade frequency (50-150/4y).
Works in bull/bear via regime adaptation. 6h timeframe balances responsiveness and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7339_6h_adx_regime_donchian_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOL_SPIKE_MULTIPLIER = 2.0
VOL_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for ADX regime
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to LTF (6h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_SPIKE_MULTIPLIER if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on ADX
        is_trending = adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = adx_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Donchian breakout levels
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Donchian retest levels (pullback to channel)
        retest_up = (close[i] >= lowest_low[i] * 0.995) and (close[i] <= lowest_low[i] * 1.005)
        retest_down = (close[i] >= highest_high[i] * 0.995) and (close[i] <= highest_high[i] * 1.005)
        
        # Enter new positions only if flat
        if position == 0:
            if is_trending:
                # Trend mode: trade breakouts in direction of momentum
                if breakout_up and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_down and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            elif is_ranging:
                # Range mode: fade at extremes with volume spike
                if retest_down and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif retest_up and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            else:
                # Transition zone: no trades
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7339_6h_adx_regime_donchian_v1
Hypothesis: 6h Donchian(20) breakouts filtered by 12h ADX regime and volume confirmation.
In trending markets (ADX > 25), trade breakouts in direction of trend.
In ranging markets (ADX < 20), fade at Donchian extremes with volume spike.
Uses 12h HTF for regime detection to avoid whipsaws. Designed for low trade frequency (50-150/4y).
Works in bull/bear via regime adaptation. 6h timeframe balances responsiveness and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7339_6h_adx_regime_donchian_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOL_SPIKE_MULTIPLIER = 2.0
VOL_MA_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for ADX regime
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h - np.roll(high_12h, 1))
    down_move = pd.Series(np.roll(low_12h, 1) - low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values / atr_12h
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False, min_periods=ADX_PERIOD).mean().values
    
    # Align ADX to LTF (6h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(adx_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_SPIKE_MULTIPLIER if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on ADX
        is_trending = adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = adx_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Donchian breakout levels
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Donchian retest levels (pullback to channel)
        retest_up = (close[i] >= lowest_low[i] * 0.995) and (close[i] <= lowest_low[i] * 1.005)
        retest_down = (close[i] >= highest_high[i] * 0.995) and (close[i] <= highest_high[i] * 1.005)
        
        # Enter new positions only if flat
        if position == 0:
            if is_trending:
                # Trend mode: trade breakouts in direction of momentum
                if breakout_up and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_down and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            elif is_ranging:
                # Range mode: fade at extremes with volume spike
                if retest_down and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif retest_up and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            else:
                # Transition zone: no trades
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals