#!/usr/bin/env python3
"""
Experiment #002: 30m Supertrend + 4h HMA + RSI Pullback with ATR Stoploss

Hypothesis: Previous 30m strategies failed due to over-complication (too many
conflicting filters) and insufficient trade generation. This strategy uses:

1. SUPERTREND (ATR=10, mult=3): Clean trend direction signal, proven in crypto
2. 4H HMA(21): Stable HTF bias filter (from best performing strategy)
3. RSI(7) pullback: Entry on dips in uptrend, rallies in downtrend
4. ATR(14) stoploss: 2.5*ATR trailing stop to protect capital

Why this should work better than #001 (Sharpe=-3.709):
- Simpler logic = fewer conflicting conditions = more trades
- Supertrend cleaner than Fisher for trend direction
- RSI pullback (not extreme) = more frequent entries
- Discrete position sizing (0.0, ±0.25) = less fee churn
- Proper stoploss tracking = limits drawdown in 2022-style crashes

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.25 base, discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target: 40-80 trades/year, Sharpe > 0 on all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_pullback_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=up, -1=down)
    """
    n = len(close)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if close[i-1] <= supertrend[i-1]:
            # Previous close below supertrend = was downtrend
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            # Previous close above supertrend = was uptrend
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
    
    return supertrend, direction

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, atr_14, 3.0)
    rsi_7 = calculate_rsi(close, 7)
    
    signals = np.zeros(n)
    
    # Position tracking for stoploss (Rule 6)
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STOPLOSS_MULT = 2.5
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            continue
        if np.isnan(rsi_7[i]):
            continue
        
        current_signal = 0.0
        
        # === TREND DIRECTION ===
        uptrend = st_direction[i] == 1
        downtrend = st_direction[i] == -1
        
        # === 4H HMA BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === STOPLOSS CHECK (before new entry) ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:  # Long position
                # Update highest price for trailing stop
                if close[i] > highest_price:
                    highest_price = close[i]
                stop_price = highest_price - STOPLOSS_MULT * atr_14[i]
                if close[i] < stop_price:
                    stoploss_triggered = True
            
            if position_side < 0:  # Short position
                # Update lowest price for trailing stop
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stop_price = lowest_price + STOPLOSS_MULT * atr_14[i]
                if close[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            # Close position on stoploss
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            current_signal = 0.0
        else:
            # === ENTRY CONDITIONS ===
            if not in_position:
                # Long: uptrend + 4h bull bias + RSI pullback to 35-55
                long_entry = uptrend and bull_bias and (35 < rsi_7[i] < 55)
                
                # Short: downtrend + 4h bear bias + RSI rally to 45-65
                short_entry = downtrend and bear_bias and (45 < rsi_7[i] < 65)
                
                if long_entry:
                    current_signal = BASE_SIZE
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_price = close[i]
                elif short_entry:
                    current_signal = -BASE_SIZE
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    lowest_price = close[i]
            else:
                # === TREND REVERSAL EXIT ===
                trend_reversal = False
                if position_side > 0 and not uptrend:
                    trend_reversal = True
                if position_side < 0 and not downtrend:
                    trend_reversal = True
                
                if trend_reversal:
                    current_signal = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    highest_price = 0.0
                    lowest_price = 0.0
                else:
                    # Hold position
                    current_signal = position_side * BASE_SIZE
                    if position_side > 0 and close[i] > highest_price:
                        highest_price = close[i]
                    if position_side < 0 and (lowest_price == 0.0 or close[i] < lowest_price):
                        lowest_price = close[i]
        
        signals[i] = current_signal
    
    return signals