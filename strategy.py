#!/usr/bin/env python3
"""
EXPERIMENT #002 - Donchian Breakout + KAMA Adaptive Trend + 4h Filter (30m primary)
=====================================================================================
Hypothesis: 30m Donchian breakouts capture momentum moves with cleaner signals than Supertrend.
KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - fast in trends, slow in chop.
4h Donchian channel provides higher timeframe trend bias.
RSI(14) filters entries to avoid chasing breakouts at extremes.

Key features:
- Primary TF: 30m
- HTF filter: 4h Donchian(20) for major trend direction
- Trend: KAMA(10,2,30) for adaptive following
- Entry: Donchian(20) breakout + RSI filter
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should work better:
- 30m captures cleaner moves than 15m (less noise)
- KAMA adapts to volatility better than fixed EMA/HMA
- Donchian breakouts are proven momentum signals
- 4h trend filter removes counter-trend trades
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_kama_4h_30m_v1"
timeframe = "30m"
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


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian for trend filter
    donchian_4h_upper, donchian_4h_lower = calculate_donchian(
        df_4h['high'].values, df_4h['low'].values, period=20
    )
    
    # Calculate 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    donchian_4h_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_4h_upper)
    donchian_4h_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_4h_lower)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
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
        if (np.isnan(donchian_4h_upper_aligned[i]) or np.isnan(donchian_4h_lower_aligned[i]) or
            np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h Donchian trend filter
        mid_4h = (donchian_4h_upper_aligned[i] + donchian_4h_lower_aligned[i]) / 2.0
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        donchian_trend = 1 if close[i] > mid_4h else -1
        
        # Combine 4h trend signals (both must agree for strong signal)
        htf_trend = 0
        if hma_trend == 1 and donchian_trend == 1:
            htf_trend = 1
        elif hma_trend == -1 and donchian_trend == -1:
            htf_trend = -1
        
        # KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # RSI filter (avoid extreme entries)
        rsi_ok_long = rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] > 30  # Not oversold
        
        # KAMA slope confirmation (simple momentum check)
        kama_slope_long = kama[i] > kama[i - 5] if i >= 5 else False
        kama_slope_short = kama[i] < kama[i - 5] if i >= 5 else False
        
        # Calculate position size (dynamic based on ATR volatility)
        atr_pct = atr[i] / close[i] * 100
        vol_adjustment = min(1.0, 0.03 / max(atr_pct, 0.01))  # Normalize to ~3% ATR
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_adjustment))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend bullish + KAMA bullish + Donchian breakout + RSI ok + KAMA slope up
        if (htf_trend == 1 and kama_trend == 1 and breakout_long and 
            rsi_ok_long and kama_slope_long):
            target_signal = position_size
        
        # Short entry: 4h trend bearish + KAMA bearish + Donchian breakout + RSI ok + KAMA slope down
        elif (htf_trend == -1 and kama_trend == -1 and breakout_short and 
              rsi_ok_short and kama_slope_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
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
                # Exit if KAMA reverses OR 4h trend breaks
                kama_reversal_long = kama_trend == -1
                kama_reversal_short = kama_trend == 1
                htf_alignment_broken = (position_side == 1 and htf_trend == -1) or \
                                       (position_side == -1 and htf_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or htf_alignment_broken:
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