#!/usr/bin/env python3
"""
EXPERIMENT #014 - Supertrend + RSI Pullback + 4h HMA Trend Filter (30m)
========================================================================
Hypothesis: 30m Supertrend(10,3) captures trend direction with ATR-based stops.
Combined with 4h HMA(21) for major trend filter and RSI(14) pullback entries,
this reduces false signals during choppy periods. Volume confirmation filters
weak breakouts. ATR trailing stop protects capital. Different from failed
strategies by using Supertrend flips (not crossovers) with strict 4h alignment.

Key features:
- Primary TF: 30m (required for this experiment)
- HTF filter: 4h HMA(21) for trend direction
- Entry: Supertrend(10,3) flip + RSI pullback (35-65 range)
- Filter: 4h trend must align with Supertrend direction
- Volume: Must be > 20-period average
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4h_filter_30m_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            if close[i - 1] > supertrend[i - 1]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    supertrend, supertrend_direction = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or np.isnan(rsi[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        hma_4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_sma[i]
        
        # Supertrend direction
        st_direction = supertrend_direction[i]
        
        # RSI pullback filter (avoid extreme entries)
        rsi_valid = 35 < rsi[i] < 65
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + 4h trend up + volume + RSI valid
        if st_direction == 1 and hma_4h_trend == 1 and volume_confirmed and rsi_valid:
            # Check for Supertrend flip (previous was bearish)
            if i > 0 and supertrend_direction[i - 1] == -1:
                target_signal = SIZE
        
        # Short entry: Supertrend bearish + 4h trend down + volume + RSI valid
        elif st_direction == -1 and hma_4h_trend == -1 and volume_confirmed and rsi_valid:
            # Check for Supertrend flip (previous was bullish)
            if i > 0 and supertrend_direction[i - 1] == 1:
                target_signal = -SIZE
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
            elif position_side != 0:
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals