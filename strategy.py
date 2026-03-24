#!/usr/bin/env python3
"""
Experiment #730: 1h Primary + 4h/1d HTF — Fisher Transform Reversal + Session Filter

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Unlike RSI which can stay extended, Fisher mean-reverts quickly making it ideal for range conditions.
Combined with 4h/1d HMA trend bias and session filter to reduce noise and fee drag.

Key innovations:
1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI in range markets
2. 4h HMA(21) for HTF trend bias - only trade with HTF direction
3. 1d HMA(50) for macro bias confirmation
4. Session filter (08-20 UTC) - trade during high liquidity hours only
5. ATR(14) 2.5x trailing stop

Entry conditions (LOOSE to ensure trades):
- LONG: 4h HMA bull + Fisher < -1.2 OR Fisher cross above -1.5 + session
- SHORT: 4h HMA bear + Fisher > +1.2 OR Fisher cross below +1.5 + session

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
Trade freq: 50-100/year (session filter reduces frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_reversal_hma_4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for identifying reversal points in range markets
    Uses (High+Low)/2 as price input, normalized over rolling period
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate median price (HL2)
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = hl2[i-period+1:i+1].max()
        lowest = hl2[i-period+1:i+1].min()
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        # Normalize to -1 to +1 range
        normalized = 2.0 * (hl2[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform formula
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
    
    return fisher, fisher_prev

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSE for trade generation) ===
        # Long: Fisher deeply oversold or crossing up from oversold
        fisher_oversold = fisher[i] < -1.2
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        
        # Short: Fisher deeply overbought or crossing down from overbought
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === ENTRY LOGIC (3+ CONFLUENCE, LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + (Fisher oversold OR cross up) + session
        # Also allow if 1d bull for stronger conviction
        if htf_4h_bull and in_session:
            if fisher_oversold or fisher_cross_up:
                if htf_1d_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + (Fisher overbought OR cross down) + session
        # Also allow if 1d bear for stronger conviction
        elif htf_4h_bear and in_session:
            if fisher_overbought or fisher_cross_down:
                if htf_1d_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals