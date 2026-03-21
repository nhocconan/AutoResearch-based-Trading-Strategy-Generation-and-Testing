#!/usr/bin/env python3
"""
EXPERIMENT #019 - Volume-Confirmed Momentum Breakout with 4h EMA Trend (15m primary)
=====================================================================================
Hypothesis: Previous strategies failed due to over-filtering (too many conditions never align).
This strategy uses SIMPLER logic: 4h EMA trend direction + 15m Donchian breakout + volume confirmation.
Key difference from failed attempts:
- Uses EMA (more stable) instead of HMA for HTF filter
- Volume confirmation instead of RSI pullback (different signal type)
- Breakout entries instead of pullback entries
- Fewer filters to ensure trades actually trigger (≥10 trades per symbol)
- Conservative position sizing (0.25 max) to control drawdown

Why this should work:
- 15m captures intraday momentum moves that 1h/4h miss
- 4h EMA(50) provides stable trend filter without whipsaw
- Volume spike confirms breakout is real (not fake breakout)
- ATR stoploss protects against reversals
- Discrete signal levels minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vol_breakout_4hema_15m_v1"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return ema


def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_volume_sma(volume, period=20):
    """Calculate Volume SMA for volume confirmation"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA(50) for trend filter
    ema_4h = calculate_ema(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    ema_15m = calculate_ema(close, 21)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative position size (25% of capital)
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_sma[i]) or atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h EMA trend filter
        price_above_4h_ema = close[i] > ema_4h_aligned[i]
        htf_trend = 1 if price_above_4h_ema else -1
        
        # Volume confirmation (current volume > 1.5x 20-period SMA)
        volume_ratio = volume[i] / vol_sma[i]
        volume_confirmed = volume_ratio > 1.5
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # EMA momentum filter (price above/below 21 EMA for confirmation)
        ema_momentum_long = close[i] > ema_15m[i]
        ema_momentum_short = close[i] < ema_15m[i]
        
        # Determine target signal based on filters
        target_signal = 0.0
        
        # Long entry: 4h uptrend + Donchian breakout + volume confirmation + EMA momentum
        if htf_trend == 1 and breakout_long and volume_confirmed and ema_momentum_long:
            target_signal = BASE_SIZE
        
        # Short entry: 4h downtrend + Donchian breakout + volume confirmation + EMA momentum
        elif htf_trend == -1 and breakout_short and volume_confirmed and ema_momentum_short:
            target_signal = -BASE_SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if HTF trend reverses OR Donchian channel breaks opposite
                htf_reversal_long = htf_trend == -1
                htf_reversal_short = htf_trend == 1
                
                # Exit if price crosses back through Donchian channel
                donchian_exit_long = close[i] < donchian_lower[i]
                donchian_exit_short = close[i] > donchian_upper[i]
                
                if htf_reversal_long or htf_reversal_short or donchian_exit_long or donchian_exit_short:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals