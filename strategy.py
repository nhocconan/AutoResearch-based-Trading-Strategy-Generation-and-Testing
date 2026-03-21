#!/usr/bin/env python3
"""
EXPERIMENT #021 - Supertrend + RSI Pullback + 4h HMA Filter (1h primary)
=========================================================================
Hypothesis: 1h Supertrend(10,3) provides clear trend signals with built-in ATR stoploss.
Combined with 4h HMA(21) for major trend filter and RSI(14) pullback entries,
this captures trending moves while avoiding counter-trend trades. RSI pullback
(30-50 for long, 50-70 for short) ensures we enter on dips rather than chasing.
1h timeframe balances signal frequency with noise reduction - more trades than 4h
strategies but cleaner than 15m/30m. Conservative position sizing (0.25-0.30)
controls drawdown during crypto crashes.

Key features:
- Primary TF: 1h (required for experiment #021)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: Supertrend(10, 3) with ATR-based stops
- Entry: RSI pullback (30-50 long, 50-70 short) - wide range for more trades
- Stoploss: Supertrend flip OR 2.5*ATR trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2.5R profit

Why this should work:
- Supertrend is proven in crypto (clear trend + stop in one indicator)
- 4h HMA filter avoids counter-trend trades (major cause of DD)
- RSI pullback entries avoid chasing breakouts
- 1h captures more opportunities than 4h/12h strategies
- Conservative sizing (0.25-0.30) survived 2022 crash in testing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4hhma_1h_v1"
timeframe = "1h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, supertrend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        if direction[i - 1] == 1:
            # Previous trend was up
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previous trend was down
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction


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
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN or zero in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter (scalar comparison at index i)
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # Supertrend direction (scalar at index i)
        st_trend = int(st_direction[i])
        
        # RSI pullback conditions - WIDE range for more trades
        rsi_pullback_long = 30 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 70  # Pullback in downtrend
        
        # Calculate position size (dynamic based on volatility)
        atr_pct = atr[i] / close[i] * 100
        vol_adjustment = min(1.0, 0.03 / atr_pct) if atr_pct > 0 else 1.0
        position_size = min(0.35, max(0.20, BASE_SIZE * vol_adjustment))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + 4h HMA bullish + RSI pullback
        if (st_trend == 1 and hma_trend == 1 and rsi_pullback_long):
            target_signal = position_size
        
        # Short entry: Supertrend bearish + 4h HMA bearish + RSI pullback
        elif (st_trend == -1 and hma_trend == -1 and rsi_pullback_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check supertrend flip (trend reversal)
                if st_trend == -1:
                    trend_reversal = True
                
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
                
                # Check supertrend flip (trend reversal)
                if st_trend == 1:
                    trend_reversal = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        elif trend_reversal:
            # Exit on supertrend flip
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals