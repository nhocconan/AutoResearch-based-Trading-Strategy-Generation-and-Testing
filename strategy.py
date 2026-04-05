#!/usr/bin/env python3
"""
Experiment #10774: 1h Donchian Breakout + 4h/1d Trend + Volume Spike
Hypothesis: 1h Donchian(20) breakouts in direction of 4h EMA20 and 1d EMA50 trend with volume confirmation
provide high-probability trend continuation trades. Works in bull markets (breakouts above EMA)
and bear markets (breakdowns below EMA). Volume filters reduce false breakouts.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Use 4h/1d for signal direction, 1h only for entry timing. Session filter: 08-20 UTC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10774_1h_donchian_breakout_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_4H_PERIOD = 20
EMA_1D_PERIOD = 50
SIGNAL_SIZE = 0.20
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
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend direction
    ema_4h = calculate_ema(df_4h['close'].values, EMA_4H_PERIOD)
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    
    # Align EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if EMAs not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: maintain current position but don't open new
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/both EMAs
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of both EMAs with volume
        long_entry = bullish_breakout and above_both_emas and volume_spike
        short_entry = bearish_breakout and below_both_emas and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3598, "output_tokens": 1473}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.007795</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3598, "output_tokens": 1473}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.007795}</parameter>duration</parameter>0.24504876136779785</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.016787</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10774: 1h Donchian Breakout + 4h/1d Trend + Volume Spike
Hypothesis: 1h Donchian(20) breakouts in direction of 4h EMA20 and 1d EMA50 trend with volume confirmation
provide high-probability trend continuation trades. Works in bull markets (breakouts above EMA)
and bear markets (breakdowns below EMA). Volume filters reduce false breakouts.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Use 4h/1d for signal direction, 1h only for entry timing. Session filter: 08-20 UTC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10774_1h_donchian_breakout_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_4H_PERIOD = 20
EMA_1D_PERIOD = 50
SIGNAL_SIZE = 0.20
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
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend direction
    ema_4h = calculate_ema(df_4h['close'].values, EMA_4H_PERIOD)
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    
    # Align EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if EMAs not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: maintain current position but don't open new
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/both EMAs
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of both EMAs with volume
        long_entry = bullish_breakout and above_both_emas and volume_spike
        short_entry = bearish_breakout and below_both_emas and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3598, "output_tokens": 1473}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.007795</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3598, "output_tokens": 1473}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.007795}</parameter>duration</parameter>0.24504876136779785</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.016787</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10774: 1h Donchian Breakout + 4h/1d Trend + Volume Spike
Hypothesis: 1h Donchian(20) breakouts in direction of 4h EMA20 and 1d EMA50 trend with volume confirmation
provide high-probability trend continuation trades. Works in bull markets (breakouts above EMA)
and bear markets (breakdowns below EMA). Volume filters reduce false breakouts.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Use 4h/1d for signal direction, 1h only for entry timing. Session filter: 08-20 UTC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10774_1h_donchian_breakout_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_4H_PERIOD = 20
EMA_1D_PERIOD = 50
SIGNAL_SIZE = 0.20
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
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend direction
    ema_4h = calculate_ema(df_4h['close'].values, EMA_4H_PERIOD)
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    
    # Align EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if EMAs not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: maintain current position but don't open new
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/both EMAs
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of both EMAs with volume
        long_entry = bullish_breakout and above_both_emas and volume_spike
        short_entry = bearish_breakout and below_both_emas and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3598, "output_tokens": 1473}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.007795</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3598, "output_tokens": 1473}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.007795}</parameter>duration</parameter>0.24504876136779785</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.016787</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10774: 1h Donchian Breakout + 4h/1d Trend + Volume Spike
Hypothesis: 1h Donchian(20) breakouts in direction of 4h EMA20 and 1d EMA50 trend with volume confirmation
provide high-probability trend continuation trades. Works in bull markets (breakouts above EMA)
and bear markets (breakdowns below EMA). Volume filters reduce false breakouts.
Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
Use 4h/1d for signal direction, 1h only for entry timing. Session filter: 08-20 UTC.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10774_1h_donchian_breakout_4h_1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
EMA_4H_PERIOD = 20
EMA_1D_PERIOD = 50
SIGNAL_SIZE = 0.20
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
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMAs for trend direction
    ema_4h = calculate_ema(df_4h['close'].values, EMA_4H_PERIOD)
    ema_1d = calculate_ema(df_1d['close'].values, EMA_1D_PERIOD)
    
    # Align EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_4H_PERIOD, EMA_1D_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if EMAs not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: maintain current position but don't open new
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/both EMAs
        above_both_emas = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        below_both_emas = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of both EMAs with volume
        long_entry = bullish_breakout and above_both_emas and volume_spike
        short_entry = bearish_breakout and below_both_emas and volume_spike
        
        #