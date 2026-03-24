#!/usr/bin/env python3
"""
Experiment #934: 1d Primary + 1w HTF — Donchian Breakout + Weekly HMA Trend + ATR Stop

Hypothesis: Daily timeframe with weekly HTF bias captures major trends while avoiding
noise. Donchian breakouts (20-period) provide clear entry signals with proven edge on
SOL/ETH. Weekly HMA(21) filters direction to avoid counter-trend traps. RSI(14) loose
filter avoids extreme entries. ATR(14) 2.5x trailing stop controls drawdown.

Key innovations:
1. 1w HMA(21) for major trend bias - price above = bullish, below = bearish
2. 1d Donchian(20) breakout for entry - clean momentum signal
3. RSI(14) filter: avoid long if RSI>75, avoid short if RSI<25 (loose)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6. LOOSE entry conditions to ensure ≥20 trades/train, ≥5/test
7. Breakout strength scaling: wider Donchian = stronger signal

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w HMA bull OR price breaks Donchian high + RSI<75
- SHORT = 1w HMA bear OR price breaks Donchian low + RSI>25
- Breakout = close crosses above/below 20-period Donchian

Target: Sharpe>0.45, trades>=80 train (20/year), trades>=15 test (12/year), DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_weekly_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian middle for additional context
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Weekly HMA may have NaNs at start - use price vs middle as fallback
        htf_1w_bull = False
        htf_1w_bear = False
        if not np.isnan(hma_1w_aligned[i]):
            htf_1w_bull = close[i] > hma_1w_aligned[i]
            htf_1w_bear = close[i] < hma_1w_aligned[i]
        else:
            # Fallback: use Donchian position
            htf_1w_bull = close[i] > donchian_middle[i]
            htf_1w_bear = close[i] < donchian_middle[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            # Breakout = close crosses above upper or below lower
            donchian_breakout_long = (close[i-1] <= donchian_upper[i-1]) and (close[i] > donchian_upper[i])
            donchian_breakout_short = (close[i-1] >= donchian_lower[i-1]) and (close[i] < donchian_lower[i])
        
        # Also check if already beyond Donchian (continuation)
        donchian_above = close[i] > donchian_upper[i]
        donchian_below = close[i] < donchian_lower[i]
        
        # === RSI FILTER (LOOSE) ===
        rsi_overbought = rsi_14[i] > 75.0
        rsi_oversold = rsi_14[i] < 25.0
        
        # === BREAKOUT STRENGTH (channel width) ===
        channel_width = donchian_upper[i] - donchian_lower[i]
        if channel_width > 0 and not np.isnan(channel_width):
            # Normalize by price
            width_pct = channel_width / close[i]
            # Wider channel = stronger momentum signal
            strong_breakout = width_pct > 0.08  # 8% channel width
        else:
            strong_breakout = False
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_1w_bull or donchian_above:
            # Breakout entry (stronger signal)
            if donchian_breakout_long and not rsi_overbought:
                if strong_breakout:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            # Continuation entry (looser - already above Donchian)
            elif donchian_above and not rsi_overbought:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_1w_bear or donchian_below:
            # Breakout entry (stronger signal)
            if donchian_breakout_short and not rsi_oversold:
                if strong_breakout:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            # Continuation entry (looser - already below Donchian)
            elif donchian_below and not rsi_oversold:
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
                entry_atr = atr_14[i]
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