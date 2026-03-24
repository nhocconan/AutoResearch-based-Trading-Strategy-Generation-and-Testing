#!/usr/bin/env python3
"""
Experiment #1071: 6h Primary + 1d/1w HTF — Donchian Breakout + RSI Pullback + HMA Bias

Hypothesis: 6h timeframe is under-explored (0 experiments). Using Donchian Channel breakouts
for trend confirmation combined with RSI pullback entries and 1d/1w HMA bias should capture
multi-day swings while avoiding whipsaws. This is simpler than regime-switching approaches
that failed in experiments #1060-1070.

Key innovations:
1. Donchian Channel(20): Breakout above 20-bar high = bullish momentum, below low = bearish
2. RSI(14) pullback: Enter on RSI 35-45 in uptrend, 55-65 in downtrend (not extremes)
3. 1d HMA(21) primary bias: Only long if price > 1d_HMA, only short if price < 1d_HMA
4. 1w HMA(21) secondary filter: Strengthens signal when aligned with 1d
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- Donchian breakout captures sustained momentum (proven on daily/weekly)
- RSI pullback entries avoid chasing breakouts (better risk/reward)
- 1d/1w HMA filters prevent counter-trend trades in strong trends
- 6h captures 2-4 day swings without 4h noise or 12h slowness
- Simpler logic = more trades (target 30-60/year, not 10-20)

Entry conditions (LOOSE to guarantee trades):
- LONG: price > 1d_HMA + Donchian breakout (close > 20-bar high) + RSI 35-55
- SHORT: price < 1d_HMA + Donchian breakdown (close < 20-bar low) + RSI 45-65
- Strengthen with 1w_HMA alignment for 0.30 size vs 0.25 base

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_rsi_hma_bias_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period
    Returns: (upper_band, lower_band, middle_band)
    Breakout above upper = bullish, below lower = bearish
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d/1w HMA alignment) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Strong alignment when both 1d and 1w agree
        strong_bull = price_above_1d and price_above_1w and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = price_below_1d and price_below_1w and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout: close above upper band (bullish momentum)
        donchian_bull = close[i] > donchian_upper[i]
        # Breakdown: close below lower band (bearish momentum)
        donchian_bear = close[i] < donchian_lower[i]
        
        # === RSI PULLBACK ZONES ===
        # In uptrend, look for RSI pullback to 35-55 zone (not oversold, just resting)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        # In downtrend, look for RSI bounce to 45-65 zone (not overbought, just resting)
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries: bullish bias + momentum + pullback
        if price_above_1d:
            if donchian_bull and rsi_pullback_long:
                if strong_bull:
                    desired_signal = SIZE_STRONG  # 0.30
                else:
                    desired_signal = SIZE_BASE  # 0.25
            # Alternative: RSI oversold in uptrend (mean reversion within trend)
            elif rsi_14[i] < 40.0 and price_above_1w:
                desired_signal = SIZE_BASE
        
        # SHORT entries: bearish bias + momentum + pullback
        if price_below_1d:
            if donchian_bear and rsi_pullback_short:
                if strong_bear:
                    desired_signal = -SIZE_STRONG  # -0.30
                else:
                    desired_signal = -SIZE_BASE  # -0.25
            # Alternative: RSI overbought in downtrend (mean reversion within trend)
            elif rsi_14[i] > 60.0 and price_below_1w:
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