#!/usr/bin/env python3
"""
Experiment #10791: 6h ADX Trend Strength + Donchian Breakout with Volume Confirmation
Hypothesis: In trending markets (ADX > 25), 6h Donchian(20) breakouts with volume confirmation
provide high-probability trend continuation. In ranging markets (ADX < 20), fade at Donchian
channels with volume divergence. Works in both bull/bear as ADX adapts to trend strength.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10791_6h_adx_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for regime filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for regime detection
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 6h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 6h indicators
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
    start = max(ADX_PERIOD, DONCHIAN_PERIOD, 20) + 1
    
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
        
        # Regime detection: trending vs ranging
        is_trending = daily_adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = daily_adx_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Mean reversion conditions (at channels)
        at_upper_channel = np.abs(close[i] - donch_upper[i]) < 0.001 * close[i] if not np.isnan(donch_upper[i]) else False
        at_lower_channel = np.abs(close[i] - donch_lower[i]) < 0.001 * close[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry logic based on regime
        if position == 0:
            if is_trending:
                # Trend following: breakout in direction with volume
                long_entry = bullish_breakout and volume_spike
                short_entry = bearish_breakout and volume_spike
                
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
            elif is_ranging:
                # Mean reversion: fade at channels with volume divergence
                # Fade when price touches channel but volume is NOT spiking (weak breakout)
                long_entry = at_lower_channel and not volume_spike
                short_entry = at_upper_channel and not volume_spike
                
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
        
        # Hold position
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3764, "output_tokens": 1740}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008992</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3764, "output_tokens": 1740}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008992}</parameter>duration</parameter>0.3192594051361084</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008992</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10791: 6h ADX Trend Strength + Donchian Breakout with Volume Confirmation
Hypothesis: In trending markets (ADX > 25), 6h Donchian(20) breakouts with volume confirmation
provide high-probability trend continuation. In ranging markets (ADX < 20), fade at Donchian
channels with volume divergence. Works in both bull/bear as ADX adapts to trend strength.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10791_6h_adx_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for regime filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for regime detection
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 6h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 6h indicators
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
    start = max(ADX_PERIOD, DONCHIAN_PERIOD, 20) + 1
    
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
        
        # Regime detection: trending vs ranging
        is_trending = daily_adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = daily_adx_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Mean reversion conditions (at channels)
        at_upper_channel = np.abs(close[i] - donch_upper[i]) < 0.001 * close[i] if not np.isnan(donch_upper[i]) else False
        at_lower_channel = np.abs(close[i] - donch_lower[i]) < 0.001 * close[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry logic based on regime
        if position == 0:
            if is_trending:
                # Trend following: breakout in direction with volume
                long_entry = bullish_breakout and volume_spike
                short_entry = bearish_breakout and volume_spike
                
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
            elif is_ranging:
                # Mean reversion: fade at channels with volume divergence
                # Fade when price touches channel but volume is NOT spiking (weak breakout)
                long_entry = at_lower_channel and not volume_spike
                short_entry = at_upper_channel and not volume_spike
                
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
        
        # Hold position
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3764, "output_tokens": 1740}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008992</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3764, "output_tokens": 1740}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008992}</parameter>duration</parameter>0.3192594051361084</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008992</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10791: 6h ADX Trend Strength + Donchian Breakout with Volume Confirmation
Hypothesis: In trending markets (ADX > 25), 6h Donchian(20) breakouts with volume confirmation
provide high-probability trend continuation. In ranging markets (ADX < 20), fade at Donchian
channels with volume divergence. Works in both bull/bear as ADX adapts to trend strength.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10791_6h_adx_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for regime filter
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily ADX for regime detection
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_adx = calculate_adx(daily_high, daily_low, daily_close, ADX_PERIOD)
    
    # Align daily ADX to 6h timeframe
    daily_adx_aligned = align_htf_to_ltf(prices, df_daily, daily_adx)
    
    # Calculate 6h indicators
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
    start = max(ADX_PERIOD, DONCHIAN_PERIOD, 20) + 1
    
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
        
        # Regime detection: trending vs ranging
        is_trending = daily_adx_aligned[i] > ADX_TREND_THRESHOLD
        is_ranging = daily_adx_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Breakout conditions
        bullish_breakout = close[i] > donch_upper[i] if not np.isnan(donch_upper[i]) else False
        bearish_breakout = close[i] < donch_lower[i] if not np.isnan(donch_lower[i]) else False
        
        # Mean reversion conditions (at channels)
        at_upper_channel = np.abs(close[i] - donch_upper[i]) < 0.001 * close[i] if not np.isnan(donch_upper[i]) else False
        at_lower_channel = np.abs(close[i] - donch_lower[i]) < 0.001 * close[i] if not np.isnan(donch_lower[i]) else False
        
        # Entry logic based on regime
        if position == 0:
            if is_trending:
                # Trend following: breakout in direction with volume
                long_entry = bullish_breakout and volume_spike
                short_entry = bearish_breakout and volume_spike
                
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
            elif is_ranging:
                # Mean reversion: fade at channels with volume divergence
                # Fade when price touches channel but volume is NOT spiking (weak breakout)
                long_entry = at_lower_channel and not volume_spike
                short_entry = at_upper_channel and not volume_spike
                
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
        
        # Hold position
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</parameter>is_last_turn</parameter>True</parameter>turns_used</parameter>1</parameter>max_turns</parameter>10</parameter>agent_type</parameter>strategist</parameter>agent_name</parameter>Strategist</parameter>has_more</parameter>False</parameter>command_stop</parameter>False</parameter>is_error</parameter>False</parameter>tool_use_remaining</parameter>59</parameter>paid_inference_attempt</parameter>0</parameter>checker_meta</parameter>{}</parameter>completion_usage</parameter>{"input_tokens": 3764, "output_tokens": 1740}</parameter>minimax_info</parameter>{}error</parameter>None</parameter>is_mcts_double_turn</parameter>False</parameter>attempt</parameter>0</parameter>prompt_id</parameter>4</parameter>cost</parameter>0.008992</parameter>parsed</parameter>{"agent_type": "strategist", "agent_name": "Strategist", "has_more": false, "command_stop": false, "is_error": false, "tool_use_remaining": 59, "paid_inference_attempt": 0, "checker_meta": {}, "completion_usage": {"input_tokens": 3764, "output_tokens": 1740}, "minimax_info": {}, "error": null, "is_mcts_double_turn": false, "attempt": 0, "prompt_id": 4, "cost": 0.008992}</parameter>duration</parameter>0.3192594051361084</parameter>label</parameter>Completed Strategist</parameter>failure_count</parameter>0</parameter>total_cost</parameter>0.008992</parameter>is_single_turn_agent</parameter>True</parameter>tool_calls</parameter>[]</parameter>output_audio</parameter>None</parameter>output_text</parameter>#!/usr/bin/env python3
"""
Experiment #10791: 6h ADX Trend Strength + Donchian Breakout with Volume Confirmation
Hypothesis: In trending markets (ADX > 25), 6h Donchian(20) breakouts with volume confirmation
provide high-probability trend continuation. In ranging markets (ADX < 20), fade at Donchian
channels with volume divergence. Works in both bull/bear as ADX adapts to trend strength.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10791_6h_adx_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate