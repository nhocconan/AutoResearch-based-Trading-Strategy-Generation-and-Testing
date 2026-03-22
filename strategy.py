#!/usr/bin/env python3
"""
Experiment #119: 12h KAMA Trend + 1d HMA Bias + RSI Pullback + ATR Stop

Hypothesis: After 118 experiments, the winning pattern is clear:
- KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA/HMA
- 12h timeframe reduces noise vs 4h while maintaining trade frequency
- 1d HMA provides stable HTF trend bias (proven in #118 Sharpe=0.478)
- RSI pullback (not extremes) captures entries during trend continuation
- ATR trailing stop protects against reversals

Why this might beat #118 (4h KAMA):
- 12h has fewer false breakouts than 4h
- KAMA efficiency ratio filters choppy periods automatically
- RSI 40-60 pullback zone = more trades than extreme RSI (avoids 0-trade failure)
- Conservative sizing (0.25/0.35) limits drawdown during 2022 crash

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_rsi_pullback_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER = trending (less smoothing), Low ER = choppy (more smoothing)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        signal = np.abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_s > 0
    rs = np.zeros(n)
    rs[mask] = gain_s[mask] / loss_s[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100  # No losses = RSI 100
    
    return rsi

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
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, period=21)
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_12h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h KAMA TREND ===
        # KAMA slope for 12h trend direction
        kama_slope_bull = kama_12h[i] > kama_12h[i - 5] if not np.isnan(kama_12h[i - 5]) else False
        kama_slope_bear = kama_12h[i] < kama_12h[i - 5] if not np.isnan(kama_12h[i - 5]) else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_12h[i]
        price_below_kama = close[i] < kama_12h[i]
        
        # === RSI PULLBACK ZONE (not extremes - ensures trades) ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_pullback_long = 40 <= rsi[i] <= 55
        # Short: RSI pulled back to 45-60 in downtrend
        rsi_pullback_short = 45 <= rsi[i] <= 60
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1d bullish + 12h KAMA bullish + RSI pullback
        if bull_trend_1d and kama_slope_bull and price_above_kama and rsi_pullback_long:
            new_signal = SIZE_STRONG
        elif bull_trend_1d and kama_slope_bull and rsi_pullback_long:
            new_signal = SIZE_BASE
        elif bull_trend_1d and price_above_kama and rsi_pullback_long:
            new_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 12h KAMA bearish + RSI pullback
        if bear_trend_1d and kama_slope_bear and price_below_kama and rsi_pullback_short:
            new_signal = -SIZE_STRONG
        elif bear_trend_1d and kama_slope_bear and rsi_pullback_short:
            new_signal = -SIZE_BASE
        elif bear_trend_1d and price_below_kama and rsi_pullback_short:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals