#!/usr/bin/env python3
"""
EXPERIMENT #017 - Donchian Breakout + 1d EMA Trend + Volume Confirmation (12h primary)
=====================================================================================
Hypothesis: 12h Donchian breakouts capture momentum moves with clear entry/exit points.
Adding 1d EMA(21) trend filter ensures we only trade with the higher timeframe direction.
Volume confirmation (volume > 1.5x 20-period avg) filters false breakouts. ATR(14) trailing
stop at 2*ATR controls downside. This should generate more trades than pure trend strategies
while maintaining directional bias from the daily chart.

Key features:
- Primary TF: 12h (REQUIRED for this experiment)
- HTF filter: 1d EMA(21) for major trend direction
- Entry: Donchian(20) breakout with volume confirmation
- Strength: Volume > 1.5x 20-period average
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work:
- 12h captures significant moves without excessive noise
- 1d EMA filter removes counter-trend trades
- Volume confirmation reduces false breakouts
- Donchian provides clear entry/exit levels
- Conservative sizing controls drawdown during crypto crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_volume_1dema_12h_v1"
timeframe = "12h"
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_volume_spike(volume, period=20):
    """Calculate volume spike indicator (current volume vs rolling average)"""
    n = len(volume)
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    
    for i in range(period - 1, n):
        if vol_avg[i] > 0:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_spike(volume, period=20)
    rsi = calculate_rsi(close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size with strong ADX
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d EMA trend filter
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        ema_trend = 1 if price_above_1d_ema else -1
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.3  # Volume > 1.3x average
        
        # ADX strength filter (only trade when ADX > 20 for 12h)
        adx_strong = adx[i] > 20
        
        # DI+ vs DI- for trend confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Calculate position size based on ADX strength (dynamic sizing)
        adx_multiplier = min(1.0 + (adx[i] - 20) / 50, 1.25)  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + 1d EMA bullish + Volume confirmed + ADX strong + DI+ > DI-
        if (breakout_long and ema_trend == 1 and volume_confirmed and 
            adx_strong and di_bullish):
            target_signal = position_size
        
        # Short entry: Donchian breakout + 1d EMA bearish + Volume confirmed + ADX strong + DI- > DI+
        elif (breakout_short and ema_trend == -1 and volume_confirmed and 
              adx_strong and di_bearish):
            target_signal = -position_size
        
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
                # Exit if 1d EMA alignment breaks OR Donchian reverses
                ema_alignment_broken = (position_side == 1 and ema_trend == -1) or \
                                       (position_side == -1 and ema_trend == 1)
                
                # Exit if price crosses opposite Donchian band
                donchian_reversal_long = close[i] < donchian_lower[i]
                donchian_reversal_short = close[i] > donchian_upper[i]
                
                if ema_alignment_broken or donchian_reversal_long or donchian_reversal_short:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals