#!/usr/bin/env python3
"""
Experiment #089: 15m Primary + 1h/1d HTF — Multi-TF Pullback with Regime Filter

Hypothesis: 15m strategies fail because (1) too many trades → fee drag, or (2) too strict → 0 trades.
SOLUTION: Use 1d HMA for major trend bias (changes rarely), 1h RSI for momentum regime,
15m only for precise pullback entries. Add Choppiness Index to avoid range whipsaws.
Session filter (00-12 UTC) reduces low-liquidity noise.

Key design:
- Timeframe: 15m (target 50-80 trades/year with strict confluence)
- HTF: 1d HMA(50) for trend bias, 1h RSI(14) for momentum
- Entry: 15m pullback to EMA(21) in direction of 1d trend + 1h RSI confirmation
- Regime: CHOP(14) < 55 = trend (take signals), CHOP > 55 = skip (range whipsaw)
- Session: 00-12 UTC only (London/NY overlap, high liquidity)
- Position size: 0.22 (conservative for 15m frequency)
- Stoploss: 2.0x ATR trailing

Why this might work on 15m:
- 1d HMA changes slowly → reduces trade frequency naturally
- 1h RSI adds momentum confirmation without being too fast
- CHOP filter avoids range markets where 15m gets chopped
- Session filter cuts 50% of bars (low volume hours)
- Pullback entries (not breakouts) have better risk/reward on lower TF

Target: Sharpe>0.20, DD>-35%, trades>=40 on train, trades>=5 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pullback_hma_rsi_chop_1h1d_session_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use 55 as threshold for filter
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    ema_15m = calculate_ema(close, period=21)
    hma_15m = calculate_hma(close, period=34)
    rsi_15m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.22  # 22% position size (conservative for 15m)
    
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
        if np.isnan(ema_15m[i]) or np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc < 12
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 55 = trending (take signals), CHOP > 55 = choppy (skip)
        is_trending = chop[i] < 55.0
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === HTF MOMENTUM (1h RSI) ===
        # RSI > 50 = bullish momentum, RSI < 50 = bearish momentum
        htf_mom_bull = rsi_1h_aligned[i] > 50.0
        htf_mom_bear = rsi_1h_aligned[i] < 50.0
        
        # === 15m TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === PULLBACK ENTRY CONDITIONS ===
        # Long: price pulls back to EMA21 but stays above it, in uptrend
        pullback_long = (close[i] > ema_15m[i]) and (close[i] < ema_15m[i] * 1.002)
        pullback_long = pullback_long and (low[i] <= ema_15m[i] * 1.001)  # touched EMA
        
        # Short: price pulls back to EMA21 but stays below it, in downtrend
        pullback_short = (close[i] < ema_15m[i]) and (close[i] > ema_15m[i] * 0.998)
        pullback_short = pullback_short and (high[i] >= ema_15m[i] * 0.999)  # touched EMA
        
        # === RSI CONFIRMATION (15m) ===
        rsi_ok_long = rsi_15m[i] > 40.0 and rsi_15m[i] < 70.0  # not oversold, not overbought
        rsi_ok_short = rsi_15m[i] > 30.0 and rsi_15m[i] < 60.0
        
        # === DESIRED SIGNAL (Multi-TF Confluence) ===
        desired_signal = 0.0
        
        # LONG: session + trend regime + 1d bull + 1h mom bull + 15m pullback + RSI ok
        if in_session and is_trending:
            if (htf_bull and htf_mom_bull and pullback_long and rsi_ok_long and hma_bull):
                desired_signal = SIZE
            # Fallback: strong 1d trend only (less strict)
            elif (htf_bull and pullback_long and rsi_15m[i] > 45.0):
                desired_signal = SIZE * 0.7
        
        # SHORT: session + trend regime + 1d bear + 1h mom bear + 15m pullback + RSI ok
            if (htf_bear and htf_mom_bear and pullback_short and rsi_ok_short and hma_bear):
                desired_signal = -SIZE
            # Fallback: strong 1d trend only (less strict)
            elif (htf_bear and pullback_short and rsi_15m[i] < 55.0):
                desired_signal = -SIZE * 0.7
        
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
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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