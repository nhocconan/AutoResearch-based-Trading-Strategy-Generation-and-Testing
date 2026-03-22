#!/usr/bin/env python3
"""
Experiment #081: 1h Supertrend with 4h HMA Trend Filter + RSI Momentum
Hypothesis: 1h timeframe is ideal for trend-following with Supertrend entries.
4h HMA provides smoother trend bias than 1d (more responsive), RSI ensures momentum.
Key insight: Previous failures had too many conflicting filters. This strategy uses:
- Supertrend(10, 3) for clear entry/exit signals on 1h
- 4h HMA(21) for trend bias (long only above, short only below)
- RSI(14) loose filter (35-65 range) to avoid extreme entries
- ATR(14) trailing stop at 2.5x for risk management
- Minimal filters to ensure trades happen (Rule 9: MUST generate trades)
Why this might work: Supertrend is proven trend-following indicator.
4h HMA is responsive enough for 1h entries but filters noise.
Loose RSI filter ensures we get trades without killing win rate.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_rsi_momentum_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, atr, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, direction (1=long, -1=short)
    """
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # If previous direction was long
            if direction[i-1] == 1:
                if close[i] > supertrend[i-1]:
                    supertrend[i] = max(upper_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    direction[i] = -1
            # If previous direction was short
            else:
                if close[i] < supertrend[i-1]:
                    supertrend[i] = min(lower_band, supertrend[i-1])
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band
                    direction[i] = 1
    
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Supertrend
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, atr, 10, 3.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
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
        
        if np.isnan(rsi[i]) or np.isnan(supertrend[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND SIGNALS ===
        supertrend_long = supertrend_dir[i] == 1
        supertrend_short = supertrend_dir[i] == -1
        
        # Supertrend flip detection (entry signal)
        supertrend_flip_long = False
        supertrend_flip_short = False
        if i > 0 and not np.isnan(supertrend_dir[i-1]):
            supertrend_flip_long = (supertrend_dir[i] == 1) and (supertrend_dir[i-1] == -1)
            supertrend_flip_short = (supertrend_dir[i] == -1) and (supertrend_dir[i-1] == 1)
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI FILTER (loose - ensure trades happen) ===
        # For longs: RSI not overbought (< 70), preferably > 35
        rsi_ok_long = 35 <= rsi[i] <= 70
        # For shorts: RSI not oversold (> 30), preferably < 65
        rsi_ok_short = 30 <= rsi[i] <= 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        
        # Path 1: Supertrend flip long + 4h trend bullish + RSI OK
        if supertrend_flip_long and bull_trend_4h:
            if rsi_ok_long:
                if ema_bullish and rsi_momentum_long:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + 4h trend bullish + price above EMA21 (trend continuation)
        if supertrend_long and bull_trend_4h:
            if close[i] > ema_21[i] and rsi[i] > 45 and rsi[i] < 65:
                new_signal = SIZE_BASE
        
        # Path 3: Simple Supertrend long with 4h trend (ensure trades happen)
        if supertrend_long:
            if bull_trend_4h or ema_bullish:
                if rsi[i] > 35 and rsi[i] < 70:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        
        # Path 1: Supertrend flip short + 4h trend bearish + RSI OK
        if supertrend_flip_short and bear_trend_4h:
            if rsi_ok_short:
                if ema_bearish and rsi_momentum_short:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + 4h trend bearish + price below EMA21 (trend continuation)
        if supertrend_short and bear_trend_4h:
            if close[i] < ema_21[i] and rsi[i] < 55 and rsi[i] > 35:
                new_signal = -SIZE_BASE
        
        # Path 3: Simple Supertrend short with 4h trend (ensure trades happen)
        if supertrend_short:
            if bear_trend_4h or ema_bearish:
                if rsi[i] > 30 and rsi[i] < 65:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals