#!/usr/bin/env python3
"""
Experiment #045: 15m Primary + 4h/1d HTF — Bollinger Mean Reversion + HMA Trend + Session Filter

Hypothesis: After 44 failed experiments, 15m strategies failed due to:
- #037, #041: 0 trades (filters too strict)
- Lower TF = more trades = fee drag kills returns

SOLUTION for 15m:
- Use 4h HMA for trend bias (not too restrictive)
- Use 1d HMA for major regime (bull/bear market detection)
- Entry: RSI(7) extreme + Bollinger Band touch + session filter (UTC 00-12)
- Choppiness Index to switch between mean-revert/trend-follow
- LOOSE filters to ensure trades generate (learned from #037/#041 zero-trade failure)
- Position size: 0.20 (smaller for 15m frequency, target 50-100 trades/year)
- Stoploss: 2.0x ATR trailing

Key design choices:
- Timeframe: 15m (high priority exploration — zero experiments so far)
- HTF: 4h HMA(21) for trend, 1d HMA(50) for regime
- Entry confluence: RSI(7)<30 or >70 + BB touch + session + HTF bias
- Regime: CHOP>55 = mean revert, CHOP<55 = trend follow
- Session: UTC 00-12 (London/NY overlap = higher volume, cleaner signals)
- Position size: 0.20 (20% of capital, conservative for 15m)
- Stoploss: 2.0x ATR (tighter for 15m swings)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_bb_rsi_hma_session_4h1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    return datetime.datetime.utcfromtimestamp(open_time / 1000).hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    
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
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
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
        
        # === SESSION FILTER (UTC 00-12 = London/NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 0 <= utc_hour <= 12
        
        # === HTF BIAS (4h HMA for trend, 1d HMA for regime) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === BOLLINGER BAND POSITION ===
        bb_width = bb_upper[i] - bb_lower[i]
        bb_position = (close[i] - bb_lower[i]) / (bb_width + 1e-10)
        
        near_lower = bb_position < 0.15  # Within 15% of lower band
        near_upper = bb_position > 0.85  # Within 15% of upper band
        touched_lower = low[i] <= bb_lower[i] * 1.002  # Touched or broke lower
        touched_upper = high[i] >= bb_upper[i] * 0.998  # Touched or broke upper
        
        # === RSI EXTREMES (LOOSE to ensure trades) ===
        rsi_oversold = rsi[i] < 35.0  # Loose threshold
        rsi_overbought = rsi[i] > 65.0  # Loose threshold
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean revert at Bollinger bands
            # LONG: near/touched lower + RSI oversold + session + 4h not strongly bear
            if (near_lower or touched_lower) and rsi_oversold and in_session and not htf_4h_bear:
                desired_signal = SIZE
            # SHORT: near/touched upper + RSI overbought + session + 4h not strongly bull
            elif (near_upper or touched_upper) and rsi_overbought and in_session and not htf_4h_bull:
                desired_signal = -SIZE
            # Fallback: extreme RSI mean reversion (ignore session for extreme moves)
            elif rsi_extreme_oversold and hma_bull:
                desired_signal = SIZE * 0.7
            elif rsi_extreme_overbought and hma_bear:
                desired_signal = -SIZE * 0.7
        else:
            # TRENDING REGIME: Follow trend with pullback entries
            # LONG: 4h bull + 1d bull + RSI pullback + session
            if htf_4h_bull and htf_1d_bull and rsi_oversold and in_session:
                desired_signal = SIZE
            # SHORT: 4h bear + 1d bear + RSI pullback + session
            elif htf_4h_bear and htf_1d_bear and rsi_overbought and in_session:
                desired_signal = -SIZE
            # Fallback: trend continuation (ignore RSI if strong trend)
            elif htf_4h_bull and htf_1d_bull and hma_bull:
                desired_signal = SIZE * 0.5
            elif htf_4h_bear and htf_1d_bear and hma_bear:
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
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