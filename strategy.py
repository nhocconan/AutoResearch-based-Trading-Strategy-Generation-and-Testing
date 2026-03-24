#!/usr/bin/env python3
"""
Experiment #130: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + ATR Volatility

Hypothesis: After 129 failed experiments, the pattern for 1h is clear:
- Session filters kill trade generation (experiments #119, #121, #125, #129 all Sharpe=0.000)
- Too many confluence filters = 0 trades (the #1 failure mode)
- RSI alone is too slow for 1h entries
- SOLUTION: Fisher Transform for faster reversal signals + loose 4h HMA bias
- Fisher Transform catches reversals in bear rallies better than RSI (proven in literature)
- 4h HMA provides trend bias without being too restrictive (close > HMA = bull bias only)
- ATR volatility filter ensures we only trade when there's movement
- LOOSE entry conditions to ensure >=40 trades/year on all symbols

Key design choices:
- Timeframe: 1h (30-60 trades/year target)
- HTF: 4h HMA(21) for trend bias, 1d HMA(50) for major trend
- Entry: Fisher Transform < -1.0 (long) or > +1.0 (short) + HTF bias
- Volatility filter: ATR(14) > 0.5% of price (avoid dead markets)
- Position size: 0.25 (25% of capital, conservative for 1h)
- Stoploss: 2.5x ATR trailing
- VERY LOOSE filters to ensure trades generate on BTC/ETH/SOL

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_atr_4h1d_v1"
timeframe = "1h"
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
    Converts price to a Gaussian normal distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate median price
    median = (high + low + close) / 3.0
    
    # Normalize price to range -1 to +1
    highest = np.zeros(n)
    lowest = np.zeros(n)
    highest[:] = np.nan
    lowest[:] = np.nan
    
    for i in range(period - 1, n):
        highest[i] = np.max(median[i-period+1:i+1])
        lowest[i] = np.min(median[i-period+1:i+1])
    
    # Normalize
    normalized = np.zeros(n)
    normalized[:] = np.nan
    for i in range(period - 1, n):
        range_hl = highest[i] - lowest[i]
        if range_hl > 1e-10:
            normalized[i] = 0.66 * ((median[i] - lowest[i]) / range_hl - 0.5) + 0.67 * normalized[i-1] if i > 0 else 0.0
            normalized[i] = np.clip(normalized[i], -0.99, 0.99)
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
        else:
            fisher[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
            trigger[i] = trigger[i-1] if not np.isnan(trigger[i-1]) else 0.0
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility filter and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index - secondary filter"""
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY FILTER (avoid dead markets) ===
        # ATR must be > 0.5% of price to ensure movement
        atr_pct = atr[i] / close[i]
        vol_ok = atr_pct > 0.005
        
        # === HTF BIAS (4h HMA) - LOOSE bias only ===
        # Only require HTF bias for entries, not for exits
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1d MAJOR TREND (very loose filter) ===
        major_bull = close[i] > hma_1d_aligned[i]
        major_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (primary entry trigger) ===
        # Long: Fisher < -1.0 (oversold) + crossing up
        fisher_oversold = fisher[i] < -1.0
        fisher_crossing_up = fisher[i] > trigger[i]
        
        # Short: Fisher > +1.0 (overbought) + crossing down
        fisher_overbought = fisher[i] > 1.0
        fisher_crossing_down = fisher[i] < trigger[i]
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === RSI FILTER (VERY LOOSE - ensure trades) ===
        rsi_not_extreme_long = rsi[i] < 70.0  # not overbought
        rsi_not_extreme_short = rsi[i] > 30.0  # not oversold
        
        # === DESIRED SIGNAL (Loose confluence for trade generation) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Fisher oversold + crossing up + vol ok + (HTF bull OR 1h HMA bull)
        if fisher_oversold and fisher_crossing_up and vol_ok:
            if htf_bull or hma_bull:
                desired_signal = SIZE
            elif major_bull:  # fallback to 1d trend
                desired_signal = SIZE * 0.7
        
        # SHORT ENTRY: Fisher overbought + crossing down + vol ok + (HTF bear OR 1h HMA bear)
        elif fisher_overbought and fisher_crossing_down and vol_ok:
            if htf_bear or hma_bear:
                desired_signal = -SIZE
            elif major_bear:  # fallback to 1d trend
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