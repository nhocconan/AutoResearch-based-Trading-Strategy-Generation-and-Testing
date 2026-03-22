#!/usr/bin/env python3
"""
Experiment #114: 1d KAMA + Supertrend + 1w HMA Trend Filter + RSI Pullback Entry

Hypothesis: After 113 experiments, daily timeframe with weekly trend filter may capture
sustained moves while avoiding whipsaws that destroyed faster timeframe strategies:
- 1d KAMA(21) adapts to volatility - smoother than EMA in ranging markets
- 1d Supertrend(ATR=10, mult=3) provides clear trend direction with built-in stop
- 1w HMA(21) gives stable higher-timeframe bias (avoid counter-trend trades)
- RSI(14) pullback entries improve timing vs pure breakout (enter on dips in uptrend)
- Conservative sizing (0.25-0.35) limits drawdown during 2022-style crashes
- 1d naturally generates 15-30 trades/year - enough for stats, few enough for low fees

Why this might beat the baseline (Sharpe=0.436):
- KAMA reduces whipsaw vs EMA/KAMA crossover strategies that failed
- Supertrend provides asymmetric risk (clear stop levels)
- 1w HMA filter prevents major counter-trend positions (critical for 2022)
- RSI pullback = better entry prices than breakout chasing
- Daily timeframe = less noise, more sustained trends

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: Supertrend + 2.5*ATR trailing (whichever hits first)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_supertrend_1w_hma_rsi_pullback_v1"
timeframe = "1d"
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

def calculate_kama(close, period=21, er_period=10, fast=2/3, slow=2/31):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + er_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = (er * (fast - slow) + slow) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    if n < period:
        return supertrend, direction
    
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period - 1] = upper_band[period - 1]
    direction[period - 1] = 1
    
    for i in range(period, n):
        # Update upper band
        if upper_band[i] > supertrend[i - 1] or close[i - 1] > supertrend[i - 1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i - 1]
        
        # Update lower band
        if lower_band[i] < supertrend[i - 1] or close[i - 1] < supertrend[i - 1]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = supertrend[i - 1]
        
        # Determine direction
        if close[i] > supertrend[i]:
            supertrend[i] = lower_band[i] if lower_band[i] > supertrend[i - 1] else supertrend[i - 1]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i] if upper_band[i] < supertrend[i - 1] else supertrend[i - 1]
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama = calculate_kama(close, 21)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # For long: RSI pulled back but still above 40 (not oversold)
        rsi_pullback_long = 40 < rsi[i] < 60
        # For short: RSI rallied but still below 60 (not overbought)
        rsi_pullback_short = 40 < rsi[i] < 60
        
        # === ATR VOLATILITY FILTER ===
        # Avoid trading when ATR is extremely low (dead market)
        atr_ok = atr[i] > 0.001 * close[i]  # ATR > 0.1% of price
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1w bullish + Supertrend bullish + KAMA bullish + RSI pullback
        if bull_trend_1w and st_bullish and kama_bullish and rsi_pullback_long and atr_ok:
            new_signal = SIZE_STRONG
        # Moderate: 1w bullish + Supertrend bullish + KAMA bullish
        elif bull_trend_1w and st_bullish and kama_bullish and atr_ok:
            new_signal = SIZE_BASE
        # Weak: Supertrend bullish + KAMA bullish (ensure trades on all symbols)
        elif st_bullish and kama_bullish and atr_ok:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1w bearish + Supertrend bearish + KAMA bearish + RSI pullback
        if bear_trend_1w and st_bearish and kama_bearish and rsi_pullback_short and atr_ok:
            new_signal = -SIZE_STRONG
        # Moderate: 1w bearish + Supertrend bearish + KAMA bearish
        elif bear_trend_1w and st_bearish and kama_bearish and atr_ok:
            new_signal = -SIZE_BASE
        # Weak: Supertrend bearish + KAMA bearish (ensure trades on all symbols)
        elif st_bearish and kama_bearish and atr_ok:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Supertrend stoploss
        if in_position and position_side > 0:
            if close[i] < supertrend[i]:
                new_signal = 0.0  # Supertrend flipped
            
            # Also track trailing high for ATR stop
            if close[i] > highest_close:
                highest_close = close[i]
            # ATR trailing stop: 2.5 * ATR below highest close
            atr_stop = highest_close - 2.5 * atr[i]
            if close[i] < atr_stop:
                new_signal = 0.0  # ATR stoploss hit
        
        if in_position and position_side < 0:
            if close[i] > supertrend[i]:
                new_signal = 0.0  # Supertrend flipped
            
            # Also track trailing low for ATR stop
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # ATR trailing stop: 2.5 * ATR above lowest close
            atr_stop = lowest_close + 2.5 * atr[i]
            if close[i] > atr_stop:
                new_signal = 0.0  # ATR stoploss hit
        
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