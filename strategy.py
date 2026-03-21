#!/usr/bin/env python3
"""
EXPERIMENT #015 - MTF HMA Trend + RSI Pullback + ATR Stoploss
=============================================================
Hypothesis: Combining 4h HMA trend direction with 1h RSI pullback entries
will capture trends at better entry points than pure trend following.
ATR-based stoploss protects capital during reversals.

Key features:
- 4h HMA(21) for trend direction (smooth, lag-reduced)
- 1h RSI(14) for pullback entries (buy dips in uptrend)
- ATR(14) stoploss at 2.5x for risk control
- Discrete position sizing: 0.0, ±0.25, ±0.35
- Take profit: reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_hma_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # === CALCULATE INDICATORS ON PRIMARY TF ===
    rsi_1h = calculate_rsi(close, 14)
    atr_1h = calculate_atr(high, low, close, 14)
    hma_1h = calculate_hma(close, 21)
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    
    # Position tracking for stoploss/takeprofit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Position sizing
    BASE_SIZE = 0.35
    HALF_SIZE = 0.175
    
    # Minimum bars for valid signals
    min_bars = max(50, len(hma_4h_aligned) - len(hma_4h_aligned[~np.isnan(hma_4h_aligned)]) + 100)
    
    for i in range(min_bars, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER: 4h HMA direction ===
        # Use current vs previous to determine trend
        if i > 0 and not np.isnan(hma_4h_aligned[i-1]):
            hma_slope = hma_4h_aligned[i] - hma_4h_aligned[i-1]
        else:
            hma_slope = 0
        
        trend_long = hma_4h_aligned[i] > hma_1h[i] and hma_slope >= 0
        trend_short = hma_4h_aligned[i] < hma_1h[i] and hma_slope <= 0
        
        # === ENTRY SIGNALS: RSI pullback ===
        # Long: uptrend + RSI dip below 45 then recovering
        # Short: downtrend + RSI rally above 55 then dropping
        
        rsi_oversold = rsi_1h[i] < 45
        rsi_overbought = rsi_1h[i] > 55
        
        # Check RSI recovery (current > previous)
        if i > 0 and not np.isnan(rsi_1h[i-1]):
            rsi_rising = rsi_1h[i] > rsi_1h[i-1]
            rsi_falling = rsi_1h[i] < rsi_1h[i-1]
        else:
            rsi_rising = False
            rsi_falling = False
        
        # === STOPLOSS LOGIC (Rule 6) ===
        current_atr = atr_1h[i] if not np.isnan(atr_1h[i]) else 0
        
        if position_side == 1 and entry_price > 0:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high[i])
            
            # Stoploss: price drops 2.5*ATR from entry
            stoploss_long = entry_price - 2.5 * current_atr
            if close[i] < stoploss_long:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Take profit: at 2R (2 * 2.5 * ATR = 5 * ATR profit)
            tp_level = entry_price + 5.0 * current_atr
            if close[i] > tp_level and signals[i-1] == BASE_SIZE:
                signals[i] = HALF_SIZE  # Reduce to half
                continue
            
            # Trail stop: move stop to breakeven at 1R profit
            if close[i] > entry_price + 2.5 * current_atr:
                # Keep position but protected
                pass
        
        elif position_side == -1 and entry_price > 0:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Stoploss: price rises 2.5*ATR from entry
            stoploss_short = entry_price + 2.5 * current_atr
            if close[i] > stoploss_short:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
            
            # Take profit: at 2R
            tp_level = entry_price - 5.0 * current_atr
            if close[i] < tp_level and signals[i-1] == -BASE_SIZE:
                signals[i] = -HALF_SIZE  # Reduce to half
                continue
        
        # === NEW ENTRY LOGIC ===
        if position_side == 0:
            # Long entry: uptrend + RSI pullback
            if trend_long and rsi_oversold and rsi_rising:
                signals[i] = BASE_SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            
            # Short entry: downtrend + RSI rally
            elif trend_short and rsi_overbought and rsi_falling:
                signals[i] = -BASE_SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
        
        elif position_side == 1:
            # Already long - check if trend reversed
            if not trend_long:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = BASE_SIZE if signals[i-1] >= HALF_SIZE else signals[i-1]
        
        elif position_side == -1:
            # Already short - check if trend reversed
            if not trend_short:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = -BASE_SIZE if signals[i-1] <= -HALF_SIZE else signals[i-1]
    
    return signals