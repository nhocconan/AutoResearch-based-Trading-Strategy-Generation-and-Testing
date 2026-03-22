#!/usr/bin/env python3
"""
Experiment #082: 4h KAMA Adaptive Trend with 1d HMA Filter + RSI Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - 
follows price closely in trends, flattens in ranges. This should reduce whipsaw
compared to fixed EMA/HMA while maintaining trade frequency.

Key insight from failures: Mean-reversion on 4h is getting destroyed. Pure trend
following with adaptive smoothing should work better. KAMA's efficiency ratio
automatically adjusts smoothing - no need for complex regime detection.

Strategy components:
- KAMA(10,2,30) on 4h: adaptive trend following (ER-based smoothing)
- 1d HMA(21): trend bias filter (long above, short below)
- RSI(14): avoid extreme entries (25-75 range, wide enough for trades)
- ATR(14) trailing stop at 2.5x for risk management
- Position sizing: 0.25 base, 0.30 strong signals (discrete levels)

Why this might beat Supertrend baseline:
- KAMA adapts to volatility automatically (no fixed multiplier like Supertrend)
- Fewer whipsaws in choppy markets while catching trends
- Simpler than complex regime-switching strategies that failed
- Proven in literature for 4h+ timeframes

Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_1d_hma_rsi_adaptive_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio (ER).
    ER = |price change| / sum of absolute price changes over period
    High ER (trending) -> fast smoothing constant
    Low ER (ranging) -> slow smoothing constant
    
    Parameters:
    - er_period: lookback for efficiency ratio (default 10)
    - fast_period: fast SC period (default 2)
    - slow_period: slow SC period (default 30)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = slow_sc + er * (fast_sc - slow_sc)
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # KAMA adaptive moving averages
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    # EMA for additional confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA CROSSOVER SIGNALS ===
        # Fast KAMA crosses above Slow KAMA = bullish
        kama_bull_cross = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        # Fast KAMA crosses below Slow KAMA = bearish
        kama_bear_cross = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # KAMA alignment (already crossed, maintaining direction)
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI FILTER (avoid extreme entries) ===
        # Wide range to ensure trades happen
        rsi_ok_long = 25 <= rsi[i] <= 75
        rsi_ok_short = 25 <= rsi[i] <= 75
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        # Price position vs KAMA
        price_above_kama = close[i] > kama_fast[i]
        price_below_kama = close[i] < kama_fast[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        
        # Path 1: KAMA bullish cross + 1d trend bullish + RSI OK (strong signal)
        if kama_bull_cross and bull_trend_1d:
            if rsi_ok_long and rsi_momentum_long:
                new_signal = SIZE_STRONG
        
        # Path 2: KAMA aligned bullish + 1d trend + price above KAMA (trend continuation)
        if kama_bullish and bull_trend_1d:
            if price_above_kama and rsi[i] > 45 and rsi[i] < 70:
                if ema_bullish:
                    new_signal = SIZE_BASE
        
        # Path 3: Simple KAMA bullish + RSI momentum (ensure trades happen)
        if kama_bullish and bull_trend_1d:
            if rsi[i] > 35 and rsi[i] < 75:
                if price_above_kama:
                    new_signal = SIZE_BASE
        
        # Path 4: KAMA cross without 1d filter but with EMA confirmation
        if kama_bull_cross:
            if ema_bullish and rsi_ok_long:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        
        # Path 1: KAMA bearish cross + 1d trend bearish + RSI OK (strong signal)
        if kama_bear_cross and bear_trend_1d:
            if rsi_ok_short and rsi_momentum_short:
                new_signal = -SIZE_STRONG
        
        # Path 2: KAMA aligned bearish + 1d trend + price below KAMA (trend continuation)
        if kama_bearish and bear_trend_1d:
            if price_below_kama and rsi[i] < 55 and rsi[i] > 30:
                if ema_bearish:
                    new_signal = -SIZE_BASE
        
        # Path 3: Simple KAMA bearish + RSI momentum (ensure trades happen)
        if kama_bearish and bear_trend_1d:
            if rsi[i] > 25 and rsi[i] < 65:
                if price_below_kama:
                    new_signal = -SIZE_BASE
        
        # Path 4: KAMA cross without 1d filter but with EMA confirmation
        if kama_bear_cross:
            if ema_bearish and rsi_ok_short:
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