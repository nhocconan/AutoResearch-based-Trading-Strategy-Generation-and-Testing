#!/usr/bin/env python3
"""
Experiment #064: 12h Primary + 1d HTF — KAMA Trend + ATR Volatility Breakout + RSI Filter

Hypothesis: After 63 failed experiments, the pattern is clear:
- Complex multi-filter strategies (CRSI+Chop+Fisher+HMA) generate 0 trades or negative Sharpe
- 6h timeframe strategies consistently failed (experiments #054-#063)
- 12h needs SIMPLER logic with fewer confluence requirements to ensure trades
- KAMA (Kaufman Adaptive Moving Average) adapts to volatility - better than HMA for range/trend switches
- ATR volatility breakout ensures entries during momentum expansion (not chop)
- 1d HMA provides major trend bias without being too restrictive
- LOOSE RSI filter (25-75) ensures we don't block entries at minor extremes
- This is SIMPLER than #064's Donchian+Chop+HMA+RSI dual-regime (too many filters = 0 trades)

Key design choices:
- Timeframe: 12h (20-50 trades/year target)
- HTF: 1d HMA(50) for major trend bias
- Entry: KAMA(21) trend + ATR(14) expansion (>1.3x avg) + RSI(14) filter
- Position size: 0.30 (30% of capital)
- Stoploss: 2.5x ATR trailing
- Fewer filters = more trades = better chance of positive Sharpe

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_atr_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    More responsive in trends, smoother in chop
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = change / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA
    kama[period] = np.mean(close[:period+1])
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, lookback=30):
    """ATR ratio - current ATR / average ATR over lookback (volatility expansion)"""
    n = len(atr)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(atr[i]):
            avg_atr = np.nanmean(atr[i-lookback+1:i+1])
            if avg_atr > 1e-10:
                ratio[i] = atr[i] / avg_atr
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, lookback=30)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === KAMA SLOPE (momentum) ===
        kama_slope_bull = kama[i] > kama[i-1] if not np.isnan(kama[i-1]) else False
        kama_slope_bear = kama[i] < kama[i-1] if not np.isnan(kama[i-1]) else False
        
        # === VOLATILITY EXPANSION (ATR ratio > 1.3 = expanding vol) ===
        vol_expansion = atr_ratio[i] > 1.3
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 25.0 and rsi[i] < 80.0
        rsi_ok_short = rsi[i] < 75.0 and rsi[i] > 20.0
        rsi_momentum_long = rsi[i] > 45.0
        rsi_momentum_short = rsi[i] < 55.0
        
        # === DESIRED SIGNAL (Simplified logic - fewer filters = more trades) ===
        desired_signal = 0.0
        
        # LONG: KAMA bull + HTF bull + RSI ok + (vol expansion OR RSI momentum)
        if kama_bull and htf_bull and rsi_ok_long:
            if vol_expansion or rsi_momentum_long:
                desired_signal = SIZE
        
        # SHORT: KAMA bear + HTF bear + RSI ok + (vol expansion OR RSI momentum)
        elif kama_bear and htf_bear and rsi_ok_short:
            if vol_expansion or rsi_momentum_short:
                desired_signal = -SIZE
        
        # Fallback: Strong KAMA trend (ignore HTF if KAMA very strong)
        if desired_signal == 0.0:
            if kama_bull and kama_slope_bull and rsi[i] > 40.0 and rsi[i] < 75.0:
                desired_signal = SIZE * 0.7
            elif kama_bear and kama_slope_bear and rsi[i] > 25.0 and rsi[i] < 60.0:
                desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals