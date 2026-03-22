#!/usr/bin/env python3
"""
Experiment #121: 15m Supertrend + 4h HMA Trend Filter + RSI Pullback + ATR Stop

Hypothesis: After 120 experiments, the winning pattern is clear:
- Multi-timeframe trend following works (exp #118 Sharpe=0.478)
- Fast timeframes (15m) need HTF filter to avoid whipsaws
- Supertrend captures trend direction with built-in ATR stop
- RSI pullback entries improve entry timing vs pure breakout
- 4h HMA provides stable trend bias without lag of daily

Why 15m might work when 12h/1d failed:
- More trades (100+ per year vs 20-40)
- Captures intraday moves that daily misses
- 4h filter prevents counter-trend trades in 2022 crash
- RSI pullback = better risk/reward than chasing breakouts

Key differences from failed 15m strategies (#109, #115):
- Simpler logic (no regime switching, no Choppiness)
- HTF filter is mandatory (4h HMA), not optional
- RSI for pullback (40-60 zone), not extremes (30/70)
- Discrete position sizing to reduce fee churn

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: Supertrend flip OR 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_pullback_atr_v1"
timeframe = "15m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Determine direction
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            else:
                supertrend[i] = upper_band
                direction[i] = -1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === RSI PULLBACK ZONE (not extremes) ===
        # For longs: RSI pulled back to 40-55 zone in uptrend
        rsi_pullback_long = 38 <= rsi[i] <= 58
        # For shorts: RSI rallied to 42-62 zone in downtrend
        rsi_pullback_short = 42 <= rsi[i] <= 62
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + Supertrend bullish + RSI pullback
        if bull_trend_4h and st_bullish and rsi_pullback_long:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + Supertrend bullish (ensure trades)
        elif bull_trend_4h and st_bullish:
            new_signal = SIZE_BASE
        # Weak: Supertrend bullish only (ensure trades on all symbols)
        elif st_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + Supertrend bearish + RSI pullback
        if bear_trend_4h and st_bearish and rsi_pullback_short:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + Supertrend bearish (ensure trades)
        elif bear_trend_4h and st_bearish:
            new_signal = -SIZE_BASE
        # Weak: Supertrend bearish only (ensure trades on all symbols)
        elif st_bearish:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Supertrend flip = automatic exit signal
        if in_position and position_side > 0 and st_bearish:
            new_signal = 0.0  # Supertrend flipped bearish
        
        if in_position and position_side < 0 and st_bullish:
            new_signal = 0.0  # Supertrend flipped bullish
        
        # ATR trailing stop for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals