#!/usr/bin/env python3
"""
Experiment #1538: 4h Primary + 1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 4h timeframe with Connors RSI (CRSI) mean reversion works well in 
crypto's frequent range markets, while 1d HMA provides trend bias to avoid 
counter-trend disasters. Choppiness Index switches between mean-reversion 
(range) and trend-following modes.

Key components:
1. 1d HMA(21) for major trend bias (never short when price >> 1d HMA)
2. 4h Choppiness Index(14) for regime: CHOP>61.8=range, CHOP<38.2=trend
3. Range regime: Connors RSI extremes (CRSI<25 long, CRSI>75 short)
4. Trend regime: KAMA(10) + Donchian(20) breakout with 1d confirmation
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- Connors RSI has 75% win rate in backtests (quantitative literature)
- LOOSE CRSI thresholds (25/75 not 10/90) guarantee sufficient trades
- 1d HMA filter prevents major counter-trend positions
- 4h TF = natural 25-40 trades/year (fee-efficient)
- Regime adaptation prevents trend strategies from dying in chop

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG range: CHOP>61 + CRSI<25 + price>1d_HMA*0.98
- SHORT range: CHOP>61 + CRSI>75 + price<1d_HMA*1.02
- LONG trend: CHOP<38 + 1d_HMA bullish + KAMA bullish + Donchian breakout
- SHORT trend: CHOP<38 + 1d_HMA bearish + KAMA bearish + Donchian breakdown

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_kama_1d_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1] if not np.isnan(kama[i - 1]) else close[i]
    
    return kama

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
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) of streak
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    streak_gain = np.insert(streak_gain, 0, 0)
    streak_loss = np.insert(streak_loss, 0, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_streak_loss != 0
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs_streak[mask]))
    
    # PercentRank(100) - percentile of current return vs last 100 returns
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        if len(window) > 0:
            count_below = np.sum(window < returns[i])
            percent_rank[i] = count_below / len(window) * 100
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
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
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(kama_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donch_upper[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        is_neutral_regime = not is_trend_regime and not is_range_regime
        
        # === TREND DIRECTION (1d HMA bias) ===
        hma_1d = hma_1d_aligned[i]
        price_above_1d = close[i] > hma_1d * 0.995  # slight buffer
        price_below_1d = close[i] < hma_1d * 1.005  # slight buffer
        
        # === KAMA TREND (4h) ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === CONNORS RSI ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 25  # LOOSE threshold for trades
        crsi_overbought = crsi_val > 75  # LOOSE threshold for trades
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            donchian_breakout_long = close[i] > donch_upper[i-1]
            donchian_breakout_short = close[i] < donch_lower[i-1]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Connors RSI mean reversion (primary strategy)
        if is_range_regime:
            # LONG: CRSI oversold + price not too far below 1d HMA
            if crsi_oversold and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + price not too far above 1d HMA
            elif crsi_overbought and price_below_1d:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Donchian breakout + KAMA + 1d bias
        elif is_trend_regime:
            # LONG: 1d bullish + KAMA bullish + Donchian breakout
            if price_above_1d and kama_bullish and donchian_breakout_long:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1d bearish + KAMA bearish + Donchian breakdown
            elif price_below_1d and kama_bearish and donchian_breakout_short:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL REGIME: CRSI extremes only with 1d confirmation
        elif is_neutral_regime:
            # LONG: CRSI very oversold + 1d bullish bias
            if crsi_val < 20 and price_above_1d:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI very overbought + 1d bearish bias
            elif crsi_val > 80 and price_below_1d:
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