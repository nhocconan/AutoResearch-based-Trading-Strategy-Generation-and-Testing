#!/usr/bin/env python3
"""
Experiment #999: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: 1h timeframe with strict regime filtering will generate fewer but higher quality
trades. Using Choppiness Index to detect range vs trend, then applying appropriate strategy
(mean reversion in range, trend follow in trend) with Connors RSI for precise entry timing.

Key innovations:
1. Choppiness Index (14): CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (trend follow)
2. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for entry timing
3. 4h HMA(21) + 12h HMA(21) dual trend filter (only trade with both HTF aligned)
4. Session filter: 08-20 UTC (high liquidity hours)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Why this should work:
- Choppiness filter prevents trend strategies in choppy markets (2022-2023 were choppy)
- CRSI catches reversals better than standard RSI (75% win rate in literature)
- Dual HTF (4h+12h) ensures we only trade when both agree (fewer but better trades)
- 1h entries with 4h/12h direction = HTF trade frequency with 1h precision
- Session filter avoids low-liquidity hours where fakeouts common

Entry conditions (LOOSE enough for trades):
- LONG = 4h HMA bull + 12h HMA bull + (CHOP>61.8 + CRSI<15 OR CHOP<38.2 + CRSI<30) + session
- SHORT = 4h HMA bear + 12h HMA bear + (CHOP>61.8 + CRSI>85 OR CHOP<38.2 + CRSI>70) + session
- Relaxed CRSI thresholds to ensure 40-80 trades/year target

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            atr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(np.abs(streak), streak_period)
    # Adjust sign based on streak direction
    streak_rsi = np.where(streak >= 0, streak_rsi, 100.0 - streak_rsi)
    
    # Percent Rank of price changes
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            pct_rank[i] = 100.0 * np.sum(returns[:-1] < current_return) / (len(returns) - 1)
    
    # CRSI
    crsi = (rsi_short + streak_rsi + pct_rank) / 3.0
    crsi[:rank_period] = np.nan
    
    return crsi

def get_hour_from_open_time(prices):
    """Extract hour from open_time for session filter"""
    try:
        # open_time is in milliseconds
        hours = (prices['open_time'].values // (1000 * 3600)) % 24
        return hours
    except Exception:
        return np.zeros(len(prices), dtype=np.int64)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Session filter (08-20 UTC = high liquidity)
    hours = get_hour_from_open_time(prices)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Need more warmup for CRSI
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h + 12h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Both HTF must agree for strong signal
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === REGIME DETECTION (Choppiness) ===
        is_choppy = choppiness[i] > 61.8  # Range market
        is_trending = choppiness[i] < 38.2  # Trend market
        
        # === CRSI ENTRY SIGNALS ===
        # Mean reversion in choppy market
        crsi_oversold_mr = crsi[i] < 25  # Relaxed from 10 for more trades
        crsi_overbought_mr = crsi[i] > 75  # Relaxed from 90 for more trades
        
        # Trend follow in trending market
        crsi_pullback_long = crsi[i] < 45  # Pullback in uptrend
        crsi_pullback_short = crsi[i] > 55  # Pullback in downtrend
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC (LOOSE THRESHOLDS FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths
        if htf_strong_bull and session_ok:
            # Path 1: Choppy market + mean reversion (CRSI oversold)
            if is_choppy and crsi_oversold_mr:
                desired_signal = SIZE_BASE
            # Path 2: Trending market + pullback entry
            elif is_trending and crsi_pullback_long:
                desired_signal = SIZE_STRONG
            # Path 3: Neutral regime + strong oversold
            elif crsi[i] < 20:
                desired_signal = SIZE_BASE
        
        # SHORT entries - multiple paths
        elif htf_strong_bear and session_ok:
            # Path 1: Choppy market + mean reversion (CRSI overbought)
            if is_choppy and crsi_overbought_mr:
                desired_signal = -SIZE_BASE
            # Path 2: Trending market + pullback entry
            elif is_trending and crsi_pullback_short:
                desired_signal = -SIZE_STRONG
            # Path 3: Neutral regime + strong overbought
            elif crsi[i] > 80:
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