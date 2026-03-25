#!/usr/bin/env python3
"""
Experiment #1314: 1d Primary + 1w HTF — Dual Regime (Choppiness + Donchian/CRSI)

Hypothesis: Daily timeframe with weekly trend bias should achieve 20-50 trades/year
with minimal fee drag. This combines proven patterns from research:

1. 1w HMA(21) for major trend bias (only trade with weekly direction)
2. Choppiness Index(14) for regime detection (>61.8 = range, <38.2 = trend)
3. Trending regime: Donchian(20) breakout with RSI(14) confirmation
4. Choppy regime: Connors RSI mean reversion (RSI2 + RSI_Streak + PercentRank)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing (0.0, ±0.25, ±0.30) to minimize fee churn

Why this should work on 1d:
- 1d natural frequency = 10-30 trades/year (lowest fee drag)
- 1w HTF filter = strong directional bias, avoids counter-trend trades
- Dual regime = adapts to market conditions (trend vs mean-revert)
- Loose thresholds within each regime = guarantees trades
- Proven on ETH (Choppiness+CRSI Sharpe +0.923) and SOL (Donchian+HMA Sharpe +0.782)

Entry logic:
- LONG trend: 1w_HMA bullish + CHOP<38.2 + Donchian breakout + RSI>50
- SHORT trend: 1w_HMA bearish + CHOP<38.2 + Donchian breakdown + RSI<50
- LONG mean-revert: 1w_HMA bullish + CHOP>61.8 + CRSI<15 + price>SMA200
- SHORT mean-revert: 1w_HMA bearish + CHOP>61.8 + CRSI>85 + price<SMA200

Target: Sharpe>0.5, trades>=20 train, trades>=3 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_donchian_crsi_1w_v1"
timeframe = "1d"
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if tr_sum > 0 and (highest - lowest) > 0:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - mean reversion indicator"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i - streak_period):i + 1]
        gains = np.sum(np.where(streak_vals > 0, streak_vals, 0))
        losses = np.abs(np.sum(np.where(streak_vals < 0, streak_vals, 0)))
        if losses > 0:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + gains / losses))
        else:
            streak_rsi[i] = 100.0
    
    # Percent Rank(100) - where today's return ranks vs last 100 days
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        if len(returns) > 0:
            today_return = returns[-1]
            rank = np.sum(returns < today_return)
            pct_rank[i] = 100.0 * rank / len(returns)
    
    # CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close)
    sma_200 = calculate_sma(close, period=200)
    hma_21 = calculate_hma(close, period=21)
    
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
    min_bars = 250  # Need enough for CRSI rank_period + SMA200
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        chop = chop_14[i]
        is_trending = chop < 38.2
        is_choppy = chop > 61.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout
        if is_trending:
            # LONG: Weekly bullish + price breaks Donchian upper + RSI>50
            if price_above_1w and close[i] > donchian_upper[i] and not np.isnan(rsi_14[i]) and rsi_14[i] > 50:
                desired_signal = SIZE_BASE
                if rsi_14[i] > 65:
                    desired_signal = SIZE_STRONG
            
            # SHORT: Weekly bearish + price breaks Donchian lower + RSI<50
            elif price_below_1w and close[i] < donchian_lower[i] and not np.isnan(rsi_14[i]) and rsi_14[i] < 50:
                desired_signal = -SIZE_BASE
                if rsi_14[i] < 35:
                    desired_signal = -SIZE_STRONG
        
        # CHOPPY REGIME: Connors RSI mean reversion
        elif is_choppy:
            if not np.isnan(crsi[i]) and not np.isnan(sma_200[i]):
                # LONG: Weekly bullish + CRSI<15 + price>SMA200
                if price_above_1w and crsi[i] < 15 and close[i] > sma_200[i]:
                    desired_signal = SIZE_BASE
                    if crsi[i] < 10:
                        desired_signal = SIZE_STRONG
                
                # SHORT: Weekly bearish + CRSI>85 + price<SMA200
                elif price_below_1w and crsi[i] > 85 and close[i] < sma_200[i]:
                    desired_signal = -SIZE_BASE
                    if crsi[i] > 90:
                        desired_signal = -SIZE_STRONG
        
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