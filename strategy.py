#!/usr/bin/env python3
"""
EXPERIMENT #085 - Supertrend + RSI Pullback + Dual HTF Trend Filter (15m primary)
==================================================================================
Hypothesis: 15m Supertrend captures short-term momentum, but generates many false signals
in chop. Adding 4h HMA for major trend direction + 1h RSI pullback filter reduces
false entries significantly. Volume confirmation ensures we trade with institutional flow.

Key features:
- Primary TF: 15m
- HTF filters: 4h HMA(21) for major trend + 1h RSI(14) for pullback entries
- Trend: Supertrend(10, 3) on 15m for entry timing
- Entry: Supertrend flip + RSI pullback (30-50 for long, 50-70 for short) + volume > avg
- Regime: 4h price above/below HMA(21) for trend bias
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, scaled by RSI extremity
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- 15m captures more opportunities than 12h while HTF filters reduce noise
- RSI pullback ensures we enter on retracements, not chasing breakouts
- Dual HTF (1h + 4h) provides stronger trend confirmation than single HTF
- Volume filter removes low-liquidity false signals
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_dualhtf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate final Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if close[i - 1] > supertrend[i - 1]:
            # Previous close above Supertrend = bullish
            if lower_band[i] < supertrend[i - 1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
            direction[i] = 1
        elif close[i - 1] < supertrend[i - 1]:
            # Previous close below Supertrend = bearish
            if upper_band[i] > supertrend[i - 1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
    
    # Detect flips (direction change)
    flips = np.zeros(n)
    for i in range(1, n):
        if direction[i] != direction[i - 1]:
            flips[i] = direction[i]  # +1 for long flip, -1 for short flip
    
    return supertrend, direction, flips, atr


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    supertrend, st_direction, st_flips, atr_15m = calculate_supertrend(high, low, close, 10, 3.0)
    rsi_15m = calculate_rsi(close, 14)
    atr_15m = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
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
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            np.isnan(supertrend[i]) or np.isnan(atr_15m[i]) or
            np.isnan(rsi_15m[i]) or np.isnan(vol_ma[i]) or
            atr_15m[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h major trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        major_trend = 1 if price_above_4h_hma else -1
        
        # 1h RSI pullback filter
        rsi_1h_val = rsi_1h_aligned[i]
        
        # RSI pullback zones (not extreme, but retracement)
        rsi_pullback_long = 35.0 <= rsi_1h_val <= 55.0  # Pullback in uptrend
        rsi_pullback_short = 45.0 <= rsi_1h_val <= 65.0  # Pullback in downtrend
        
        # Volume confirmation (above average)
        volume_confirmed = volume[i] > vol_ma[i] * 1.0
        
        # Supertrend flip signals
        supertrend_long_flip = st_flips[i] == 1  # Flipped to bullish
        supertrend_short_flip = st_flips[i] == -1  # Flipped to bearish
        
        # Current Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Calculate position size based on RSI extremity
        rsi_multiplier = 1.0
        if position_side == 0:
            # Scale size by how deep the pullback is
            if major_trend == 1 and rsi_pullback_long:
                rsi_multiplier = 1.0 + (55.0 - rsi_1h_val) / 40.0  # Deeper pullback = larger size
            elif major_trend == -1 and rsi_pullback_short:
                rsi_multiplier = 1.0 + (rsi_1h_val - 45.0) / 40.0
            rsi_multiplier = min(1.25, max(0.8, rsi_multiplier))
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * rsi_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend flip + 4h bullish + 1h RSI pullback + volume
        if (supertrend_long_flip and major_trend == 1 and 
            rsi_pullback_long and volume_confirmed):
            target_signal = position_size
        
        # Short entry: Supertrend flip + 4h bearish + 1h RSI pullback + volume
        elif (supertrend_short_flip and major_trend == -1 and 
              rsi_pullback_short and volume_confirmed):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr_15m[i]
                
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
                trailing_stop = lowest_since_entry + 2.0 * atr_15m[i]
                
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
                entry_atr = atr_15m[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend flips against position OR 4h trend breaks
                supertrend_reversal = (position_side == 1 and st_bearish) or \
                                      (position_side == -1 and st_bullish)
                hma_trend_broken = (position_side == 1 and major_trend == -1) or \
                                   (position_side == -1 and major_trend == 1)
                
                if supertrend_reversal or hma_trend_broken:
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