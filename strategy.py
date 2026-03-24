#!/usr/bin/env python3
"""
Experiment #114: 4h Primary + 1d HTF — Donchian Breakout + KAMA Trend + Loose RSI

Hypothesis: After analyzing 100+ failed experiments, the clearest pattern is:
- Experiment #113 (1d Donchian+HMA+RSI) got Sharpe=0.233, +88.8% return — DONCHIAN WORKS
- Experiments 105-112 all got Sharpe=0.000 (ZERO TRADES) — too many filters
- Current best (Sharpe=0.351) uses 4h KAMA + RSI + BB — proven combination
- Key insight: DONCHIAN breakouts capture momentum, KAMA filters false breakouts

This strategy combines the BEST elements from successful experiments:
1. 1d KAMA = major trend bias (price above/below) — from current best
2. 4h Donchian(20) breakout = entry trigger — from #113 which worked
3. 4h KAMA(21) = trend confirmation (price above KAMA for longs) — proven filter
4. RSI(14) loose filter (>30 long, <70 short) — ensures trades on all symbols
5. ATR(14) trailing stoploss (2.5x) — risk management from current best

Key design choices:
- Timeframe: 4h (proven, 20-50 trades/year target)
- HTF: 1d for trend bias (matches #113 success pattern)
- Donchian(20): captures 20-bar breakouts (standard momentum)
- KAMA(21): adaptive trend filter (better than EMA/HMA in chop)
- RSI thresholds: 30/70 (looser than standard 20/80, ensures trades)
- Position size: 0.27 (27% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (proven in current best)

Why this should beat Sharpe=0.351:
- Donchian breakout captures strong momentum moves (like #113)
- KAMA filter reduces false breakouts in choppy markets
- 1d HTF bias prevents counter-trend entries (major improvement)
- Loose RSI ensures >=30 trades on train, >=3 on test (avoiding 0-trade failure)
- Simpler than failed experiments 105-112 (fewer confluence requirements)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_rsi_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close[period:] - close[:-period])
    sum_price_change = np.zeros(n - period)
    for i in range(n - period):
        sum_price_change[i] = np.sum(np.abs(np.diff(close[i:i+period+1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(period, n):
        if sum_price_change[i-period] > 1e-10:
            er[i] = price_change[i-period] / sum_price_change[i-period]
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period+1])
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel: Upper = highest high(period), Lower = lowest low(period)
    Breakout above upper = bullish, below lower = bearish
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.27  # 27% position size (conservative for 4h)
    
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
        if np.isnan(kama_4h[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        # Price above 1d KAMA = bull bias, below = bear bias
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 4h TREND (KAMA) ===
        # Price above 4h KAMA = bull trend, below = bear trend
        trend_bull = close[i] > kama_4h[i]
        trend_bear = close[i] < kama_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        # Break above upper = long signal, break below lower = short signal
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI > 30 (not extremely oversold)
        # For shorts: RSI < 70 (not extremely overbought)
        rsi_ok_long = rsi[i] > 30.0
        rsi_ok_short = rsi[i] < 70.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h trend bull + Donchian breakout + RSI > 30
        # SHORT: 1d bear + 4h trend bear + Donchian breakout + RSI < 70
        desired_signal = 0.0
        
        if htf_bull and trend_bull and breakout_long and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and trend_bear and breakout_short and rsi_ok_short:
            desired_signal = -SIZE
        
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