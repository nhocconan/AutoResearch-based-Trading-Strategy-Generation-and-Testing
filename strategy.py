#!/usr/bin/env python3
"""
Experiment #10783: 4h Donchian Breakout + 12h Trend + Volume Spike
Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend direction,
filtered by volume spikes, provide high-probability trend continuation trades.
Works in bull markets (breakouts above 12h EMA) and bear markets (breakdowns below 12h EMA).
Volume filters reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10783_4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        above_ema_12h = close[i] > ema_12h_aligned[i]
        below_ema_12h = close[i] < ema_12h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 12h trend with volume
        long_entry = bullish_breakout and above_ema_12h and volume_spike
        short_entry = bearish_breakout and below_ema_12h and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3337, "output_tokens": 1703}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008406</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3337, "output_tokens": 1703}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008406}</parameter>duration</parameter>0.2826247215270996</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008406</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10783: 4h Donchian Breakout + 12h Trend + Volume Spike
Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend direction,
filtered by volume spikes, provide high-probability trend continuation trades.
Works in bull markets (breakouts above 12h EMA) and bear markets (breakdowns below 12h EMA).
Volume filters reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10783_4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        above_ema_12h = close[i] > ema_12h_aligned[i]
        below_ema_12h = close[i] < ema_12h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 12h trend with volume
        long_entry = bullish_breakout and above_ema_12h and volume_spike
        short_entry = bearish_breakout and below_ema_12h and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3337, "output_tokens": 1703}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008406</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3337, "output_tokens": 1703}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008406}</parameter>duration</parameter>0.2826247215270996</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008406</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10783: 4h Donchian Breakout + 12h Trend + Volume Spike
Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend direction,
filtered by volume spikes, provide high-probability trend continuation trades.
Works in bull markets (breakouts above 12h EMA) and bear markets (breakdowns below 12h EMA).
Volume filters reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10783_4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        above_ema_12h = close[i] > ema_12h_aligned[i]
        below_ema_12h = close[i] < ema_12h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 12h trend with volume
        long_entry = bullish_breakout and above_ema_12h and volume_spike
        short_entry = bearish_breakout and below_ema_12h and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3337, "output_tokens": 1703}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008406</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3337, "output_tokens": 1703}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008406}</parameter>duration</parameter>0.2826247215270996</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008406</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10783: 4h Donchian Breakout + 12h Trend + Volume Spike
Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend direction,
filtered by volume spikes, provide high-probability trend continuation trades.
Works in bull markets (breakouts above 12h EMA) and bear markets (breakdowns below 12h EMA).
Volume filters reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10783_4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h indicators
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 12h EMA
        above_ema_12h = close[i] > ema_12h_aligned[i]
        below_ema_12h = close[i] < ema_12h_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout in direction of 12h trend with volume
        long_entry = bullish_breakout and above_ema_12h and volume_spike
        short_entry = bearish_breakout and below_ema_12h and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3337, "output_tokens": 1703}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008406</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3337, "output_tokens": 1703}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008406}</parameter>duration</parameter>0.2826247215270996</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008406</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10783: 4h Donchian Breakout + 12h Trend + Volume Spike
Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend direction,
filtered by volume spikes, provide high-probability trend continuation trades.
Works in bull markets (breakouts above 12h EMA) and bear markets (breakdowns below 12h EMA).
Volume filters reduce false breakouts. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10783_4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min