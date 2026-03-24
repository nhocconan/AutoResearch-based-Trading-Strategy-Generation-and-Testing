#!/usr/bin/env python3
"""
Experiment #137: 15m Primary + 4h/1d HTF — Camarilla Pivot Mean Reversion + Trend Filter

Hypothesis: After 120+ failed experiments, 15m strategies fail because:
1. Entry conditions too strict → 0 trades generated (Sharpe=0.000)
2. Session filters don't work on 24/7 crypto markets
3. Too many confluence requirements = never all agree

This strategy uses:
- Camarilla pivot levels from 1d HTF (proven intraday mean-reversion)
- 4h HMA(21) for soft trend bias (not hard filter)
- RSI(7) extremes for entry timing (loose: <30/>70, not <20/>80)
- ATR(14) trailing stop at 2.2x
- Position size: 0.22 (conservative for 15m frequency)

Key design for trade generation:
- RSI(7) < 30 OR price < S3 = long trigger (OR not AND)
- RSI(7) > 70 OR price > R3 = short trigger
- 4h HMA as soft bias only (70% size if against trend)
- No session filter (crypto trades 24/7)
- Target: 50-100 trades/year on 15m

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_hma_4h1d_v1"
timeframe = "15m"
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

def calculate_camarilla_pivots(df_1d):
    """
    Camarilla Pivot Levels from daily data
    R3/S3 = mean reversion levels (fade)
    R4/S4 = breakout levels (follow)
    
    Formula:
    Range = High_prev - Low_prev
    R3 = Close_prev + Range * 1.1/12
    S3 = Close_prev - Range * 1.1/12
    R4 = Close_prev + Range * 1.1/6
    S4 = Close_prev - Range * 1.1/6
    """
    n_1d = len(df_1d)
    if n_1d < 2:
        return None, None, None, None
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3 = np.zeros(n_1d)
    s3 = np.zeros(n_1d)
    r4 = np.zeros(n_1d)
    s4 = np.zeros(n_1d)
    r3[:] = np.nan
    s3[:] = np.nan
    r4[:] = np.nan
    s4[:] = np.nan
    
    for i in range(1, n_1d):
        range_hl = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        
        r3[i] = close_prev + range_hl * 1.1 / 12.0
        s3[i] = close_prev - range_hl * 1.1 / 12.0
        r4[i] = close_prev + range_hl * 1.1 / 6.0
        s4[i] = close_prev - range_hl * 1.1 / 6.0
    
    return r3, s3, r4, s4

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
    
    # Calculate Camarilla pivots from 1d data
    r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla_pivots(df_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate primary (15m) indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.22  # 22% position size (conservative for 15m)
    SIZE_REDUCED = 0.15  # Reduced size when against HTF trend
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === CAMARILLA LEVELS ===
        price_below_s3 = close[i] < s3_aligned[i]
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s4 = close[i] < s4_aligned[i]
        price_above_r4 = close[i] > r4_aligned[i]
        
        # === RSI EXTREMES (LOOSE for trade generation) ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # === ENTRY LOGIC (OR conditions to ensure trades) ===
        desired_signal = 0.0
        use_reduced_size = False
        
        # LONG: RSI oversold OR price below S3 (mean reversion)
        if rsi_oversold or price_below_s3:
            if htf_bull:
                desired_signal = SIZE
            else:
                desired_signal = SIZE_REDUCED
                use_reduced_size = True
        
        # SHORT: RSI overbought OR price above R3 (mean reversion)
        elif rsi_overbought or price_above_r3:
            if htf_bear:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE_REDUCED
                use_reduced_size = True
        
        # BREAKOUT: Price breaks R4/S4 (trend follow, higher conviction)
        if price_above_r4 and htf_bull:
            desired_signal = SIZE
            use_reduced_size = False
        elif price_below_s4 and htf_bear:
            desired_signal = -SIZE
            use_reduced_size = False
        
        # === STOPLOSS CHECK (Trailing ATR 2.2x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.2 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.2 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE_REDUCED * 0.85:
            final_signal = SIZE_REDUCED
        elif desired_signal <= -SIZE_REDUCED * 0.85:
            final_signal = -SIZE_REDUCED
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