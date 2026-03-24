#!/usr/bin/env python3
"""
Experiment #381: 4h Primary + 1d HTF — Simplified KAMA + CRSI + Donchian Breakout

Hypothesis: Previous dual-regime strategies over-filtered entries (0 trades in #375, #378, #380).
This strategy uses SIMPLIFIED regime detection with relaxed CRSI thresholds to ensure
trade generation while maintaining quality. Key innovations:

1. KAMA (Kaufman Adaptive MA) for primary trend - adapts to volatility better than HMA/EMA
2. Relaxed CRSI thresholds: <30 for long, >70 for short (not 10/90 which rarely trigger)
3. Single HTF (1d) for bias - dual HTF (1d+1w) caused over-filtering in failed experiments
4. Donchian breakout for trend continuation - proven on 4h timeframe
5. ADX hysteresis: enter at 25, exit at 18 - prevents whipsaw in transition zones
6. Position size 0.30 discrete - balances return vs drawdown, minimizes fee churn

Target: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_donchian_1d_simplified_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - smooth in trends, flat in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change = absolute price change over er_period
    change = np.abs(close_s.diff(er_period))
    
    # Sum of individual changes (volatility/noise)
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (ER) = signal / noise
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0.5).values
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Relaxed thresholds for crypto: <30 oversold, >70 overbought
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's smoothing
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Calculate and align HTF KAMA for bias (1d)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX hysteresis tracking
    prev_adx = 20.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        
        # === HTF BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        price_above_kama_4h = close[i] > kama_4h[i]
        price_below_kama_4h = close[i] < kama_4h[i]
        
        # === ADX with Hysteresis (enter 25, exit 18) ===
        adx_current = adx_14[i]
        is_trending = adx_current > 25.0 or (prev_adx > 25.0 and adx_current > 18.0)
        prev_adx = adx_current
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        long_bias = price_above_kama_1d  # HTF bullish
        
        # Entry trigger 1: Donchian breakout in trend
        breakout_long = close[i] > donchian_upper[i-1]
        
        # Entry trigger 2: CRSI oversold pullback
        crsi_oversold = crsi[i] < 30.0
        
        # Entry trigger 3: Price above 4h KAMA (momentum confirmation)
        momentum_long = price_above_kama_4h
        
        # LONG ENTRY: Need HTF bias + (breakout OR oversold pullback)
        if long_bias:
            if is_trending and breakout_long and momentum_long:
                # Trend breakout long
                desired_signal = BASE_SIZE
            elif not is_trending and crsi_oversold:
                # Range mean-reversion long
                desired_signal = BASE_SIZE
            elif crsi_oversold and momentum_long:
                # Pullback in uptrend
                desired_signal = BASE_SIZE
        
        # SHORT SETUP
        short_bias = price_below_kama_1d  # HTF bearish
        
        # Entry trigger 1: Donchian breakdown in trend
        breakout_short = close[i] < donchian_lower[i-1]
        
        # Entry trigger 2: CRSI overbought rally
        crsi_overbought = crsi[i] > 70.0
        
        # Entry trigger 3: Price below 4h KAMA (momentum confirmation)
        momentum_short = price_below_kama_4h
        
        # SHORT ENTRY: Need HTF bias + (breakdown OR overbought rally)
        if short_bias:
            if is_trending and breakout_short and momentum_short:
                # Trend breakdown short
                desired_signal = -BASE_SIZE
            elif not is_trending and crsi_overbought:
                # Range mean-reversion short
                desired_signal = -BASE_SIZE
            elif crsi_overbought and momentum_short:
                # Rally in downtrend
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 35:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_kama_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_kama_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_kama_1d:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_kama_1d:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals