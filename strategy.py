#!/usr/bin/env python3
"""
Experiment #008: 4h Primary + 12h HTF — Fisher Transform + Choppiness + HMA Trend

Hypothesis: After 7 failed experiments, the pattern shows:
- RSI-based strategies fail on BTC/ETH in bear markets (whipsaw at extremes)
- Pure trend following gets destroyed in 2022 crash and 2025 bear
- Fisher Transform catches reversals better than RSI in bear/range markets
- 4h timeframe balances trade frequency (20-50/year) vs fee drag
- 12h HMA provides trend bias without being too slow
- Choppiness Index regime switch prevents trend-following in chop

Key design choices:
- Timeframe: 4h (target 20-50 trades/year, proven to work)
- HTF: 12h HMA(21) for major trend bias
- Entry: Fisher Transform crosses + Choppiness regime filter
- Regime: CHOP>55 = range (mean revert), CHOP<55 = trend (breakout follow)
- Position size: 0.28 (28% of capital, conservative)
- Stoploss: 2.5x ATR trailing stop
- Loose Fisher thresholds (-1.5/+1.5) to ensure trades generate

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_hma_12h_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for better reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Calculate highest and lowest median over period
        highest_median = np.max(median)
        lowest_median = np.min(median)
        
        # Calculate over lookback period
        medians = []
        for j in range(i - period + 1, i + 1):
            med = (high[j] + low[j]) / 2.0
            medians.append(med)
        
        highest_median = np.max(medians)
        lowest_median = np.min(medians)
        
        range_hl = highest_median - lowest_median
        if range_hl < 1e-10:
            fisher[i] = 0.0
            if i > 0:
                fisher_prev[i] = fisher[i-1]
            continue
        
        # Normalize price
        normalized = (median - lowest_median) / range_hl
        normalized = np.clip(normalized, 0.001, 0.999)  # avoid log(0)
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_val
        else:
            fisher[i] = 0.7 * fisher[i-1] + 0.3 * fisher_val
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 55 as threshold for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index for additional filter"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Fisher reversals with trend bias
            # LONG: Fisher long + HTF bull + HMA bull + RSI ok
            if fisher_long and htf_bull and hma_bull and rsi_ok_long:
                desired_signal = SIZE
            # SHORT: Fisher short + HTF bear + HMA bear + RSI ok
            elif fisher_short and htf_bear and hma_bear and rsi_ok_short:
                desired_signal = -SIZE
            # Fallback: Fisher signal + HMA only (ignore HTF if strong)
            elif fisher_long and hma_bull and rsi[i] > 30.0:
                desired_signal = SIZE * 0.7
            elif fisher_short and hma_bear and rsi[i] < 70.0:
                desired_signal = -SIZE * 0.7
        else:
            # CHOPPY REGIME: Mean revert with Fisher extremes
            # LONG: Fisher very oversold + RSI oversold
            if fisher[i] < -2.0 and rsi_oversold and not htf_bear:
                desired_signal = SIZE
            # SHORT: Fisher very overbought + RSI overbought
            elif fisher[i] > 2.0 and rsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Fallback: Fisher cross in chop
            elif fisher_long and rsi[i] < 50.0:
                desired_signal = SIZE * 0.7
            elif fisher_short and rsi[i] > 50.0:
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