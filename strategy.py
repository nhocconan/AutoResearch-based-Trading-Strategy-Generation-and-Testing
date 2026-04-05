#!/usr/bin/env python3
"""
Experiment #10833: 12h Donchian Breakout + 1d ADX Trend + Volume Spike
Hypothesis: 12h Donchian(20) breakouts confirmed by 1d ADX > 25 and volume spike
provide high-probability trend continuation trades. ADX filters out choppy markets,
reducing false breakouts. Works in trending markets (both bull and bear) by only
trading when ADX indicates strong trend. Volume ensures breakout conviction.
Target: 75-200 total trades over 4 years (19-50/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10833_12h_donchian_breakout_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_dx(high, low, close, period):
    """Calculate Directional Index (DX) for ADX"""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_sum = WilderSmooth(tr, period)
    dm_plus_sum = WilderSmooth(dm_plus, period)
    dm_minus_sum = WilderSmooth(dm_minus, period)
    
    # Avoid division by zero
    di_plus = np.where(tr_sum > 0, (dm_plus_sum / tr_sum) * 100, 0)
    di_minus = np.where(tr_sum > 0, (dm_minus_sum / tr_sum) * 100, 0)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    return dx

def calculate_adx(high, low, close, period):
    """Calculate ADX (smoothed DX)"""
    dx = calculate_dx(high, low, close, period)
    # Smooth DX with Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    adx = WilderSmooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for trend strength filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 12h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 12h indicators
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
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily ADX not available
        if np.isnan(daily_adx_aligned[i]):
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
        
        # Trend strength filter: ADX > threshold
        strong_trend = daily_adx_aligned[i] > ADX_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout with strong trend and volume
        long_entry = bullish_breakout and strong_trend and volume_spike
        short_entry = bearish_breakout and strong_trend and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 4032, "output_tokens": 1960}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>5</parameter>cost</parameter>0.009648</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 4032, "output_tokens": 1960}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 5, "cost": 0.009648}</parameter>duration</parameter>0.34988011360168457</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.009648</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10833: 12h Donchian Breakout + 1d ADX Trend + Volume Spike
Hypothesis: 12h Donchian(20) breakouts confirmed by 1d ADX > 25 and volume spike
provide high-probability trend continuation trades. ADX filters out choppy markets,
reducing false breakouts. Works in trending markets (both bull and bear) by only
trading when ADX indicates strong trend. Volume ensures breakout conviction.
Target: 75-200 total trades over 4 years (19-50/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10833_12h_donchian_breakout_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_dx(high, low, close, period):
    """Calculate Directional Index (DX) for ADX"""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_sum = WilderSmooth(tr, period)
    dm_plus_sum = WilderSmooth(dm_plus, period)
    dm_minus_sum = WilderSmooth(dm_minus, period)
    
    # Avoid division by zero
    di_plus = np.where(tr_sum > 0, (dm_plus_sum / tr_sum) * 100, 0)
    di_minus = np.where(tr_sum > 0, (dm_minus_sum / tr_sum) * 100, 0)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    return dx

def calculate_adx(high, low, close, period):
    """Calculate ADX (smoothed DX)"""
    dx = calculate_dx(high, low, close, period)
    # Smooth DX with Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    adx = WilderSmooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for trend strength filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 12h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 12h indicators
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
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily ADX not available
        if np.isnan(daily_adx_aligned[i]):
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
        
        # Trend strength filter: ADX > threshold
        strong_trend = daily_adx_aligned[i] > ADX_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout with strong trend and volume
        long_entry = bullish_breakout and strong_trend and volume_spike
        short_entry = bearish_breakout and strong_trend and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 4032, "output_tokens": 1960}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>5</parameter>cost</parameter>0.009648</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 4032, "output_tokens": 1960}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 5, "cost": 0.009648}</parameter>duration</parameter>0.34988011360168457</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.009648</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10833: 12h Donchian Breakout + 1d ADX Trend + Volume Spike
Hypothesis: 12h Donchian(20) breakouts confirmed by 1d ADX > 25 and volume spike
provide high-probability trend continuation trades. ADX filters out choppy markets,
reducing false breakouts. Works in trending markets (both bull and bear) by only
trading when ADX indicates strong trend. Volume ensures breakout conviction.
Target: 75-200 total trades over 4 years (19-50/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10833_12h_donchian_breakout_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_dx(high, low, close, period):
    """Calculate Directional Index (DX) for ADX"""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_sum = WilderSmooth(tr, period)
    dm_plus_sum = WilderSmooth(dm_plus, period)
    dm_minus_sum = WilderSmooth(dm_minus, period)
    
    # Avoid division by zero
    di_plus = np.where(tr_sum > 0, (dm_plus_sum / tr_sum) * 100, 0)
    di_minus = np.where(tr_sum > 0, (dm_minus_sum / tr_sum) * 100, 0)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    return dx

def calculate_adx(high, low, close, period):
    """Calculate ADX (smoothed DX)"""
    dx = calculate_dx(high, low, close, period)
    # Smooth DX with Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    adx = WilderSmooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for trend strength filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 12h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 12h indicators
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
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if daily ADX not available
        if np.isnan(daily_adx_aligned[i]):
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
        
        # Trend strength filter: ADX > threshold
        strong_trend = daily_adx_aligned[i] > ADX_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry conditions: breakout with strong trend and volume
        long_entry = bullish_breakout and strong_trend and volume_spike
        short_entry = bearish_breakout and strong_trend and volume_spike
        
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

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 4032, "output_tokens": 1960}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>5</parameter>cost</parameter>0.009648</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 4032, "output_tokens": 1960}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 5, "cost": 0.009648}</parameter>duration</parameter>0.34988011360168457</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.009648</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10833: 12h Donchian Breakout + 1d ADX Trend + Volume Spike
Hypothesis: 12h Donchian(20) breakouts confirmed by 1d ADX > 25 and volume spike
provide high-probability trend continuation trades. ADX filters out choppy markets,
reducing false breakouts. Works in trending markets (both bull and bear) by only
trading when ADX indicates strong trend. Volume ensures breakout conviction.
Target: 75-200 total trades over 4 years (19-50/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10833_12h_donchian_breakout_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return