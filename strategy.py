#!/usr/bin/env python3
"""
Experiment #022: 4h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + ADX Trend

Hypothesis: After 21 failed experiments, the clearest pattern is:
- Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails on SOL (strong trends)
- SOLUTION: Dual-regime with Choppiness Index to switch between trend/mean-revert
- Connors RSI (CRSI) has proven 75% win rate for mean reversion entries
- 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
- 1d HMA provides major trend bias, 1w HMA provides macro regime filter
- ADX confirms trend strength before entering trend-following trades
- LOOSE entry conditions ensure >=30 trades on train, >=3 on test for ALL symbols

Key design choices:
- Timeframe: 4h (proven best for crypto perpetuals)
- HTF: 1d HMA for trend bias, 1w HMA for macro regime
- Entry: Connors RSI extremes + Choppiness regime + ADX confirmation
- Regime: CHOP>55 = range (mean revert), CHOP<45 = trend (breakout follow)
- Position size: 0.28 (28% of capital, discrete levels)
- Stoploss: 2.5x ATR trailing (signal→0 when hit)
- LOOSE filters: RSI 25-75 range, ADX>20 (not 25+) to ensure trades generate

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_adx_regime_1d1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - proven 75% win rate for mean reversion
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    streak[0] = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        abs_streak = np.abs(streak[max(0, i-streak_period):i+1])
        if len(abs_streak) >= streak_period:
            avg_streak = np.mean(abs_streak)
            streak_rsi[i] = min(100.0, avg_streak * 50.0)
    
    # Percent Rank component
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period):i+1])
        if len(returns) > 0 and returns[-1] != 0:
            pct_rank[i] = np.sum(returns[:-1] < returns[-1]) / len(returns[:-1]) * 100.0
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100.0
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100.0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100.0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    We use 55 as threshold for regime switch
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (discrete levels)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(crsi[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong HTF bias when both agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45-55 = transition (use both strategies)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        is_transition = not is_choppy and not is_trending
        
        # === TREND STRENGTH (ADX) ===
        adx_strong = adx[i] > 20.0  # LOOSE threshold to ensure trades
        adx_weak = adx[i] <= 20.0
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0  # LOOSE threshold
        crsi_overbought = crsi[i] > 75.0  # LOOSE threshold
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI FILTER (Backup for CRSI) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending and adx_strong:
            # TREND REGIME: Follow HTF bias with ADX confirmation
            # LONG: HTF bull + HMA bull + ADX strong
            if htf_strong_bull and hma_bull and adx_strong:
                desired_signal = SIZE
            elif htf_1d_bull and hma_bull and adx_strong and rsi[i] > 40.0:
                desired_signal = SIZE * 0.7
            # SHORT: HTF bear + HMA bear + ADX strong
            elif htf_strong_bear and hma_bear and adx_strong:
                desired_signal = -SIZE
            elif htf_1d_bear and hma_bear and adx_strong and rsi[i] < 60.0:
                desired_signal = -SIZE * 0.7
        
        elif is_choppy or adx_weak:
            # CHOPPY REGIME: Mean reversion with CRSI
            # LONG: CRSI oversold + HTF not strongly bear
            if crsi_oversold and not htf_strong_bear:
                desired_signal = SIZE
            elif crsi_extreme_oversold:
                desired_signal = SIZE  # Enter even against HTF on extreme
            elif rsi_oversold and hma_bull and not htf_strong_bear:
                desired_signal = SIZE * 0.7
            # SHORT: CRSI overbought + HTF not strongly bull
            elif crsi_overbought and not htf_strong_bull:
                desired_signal = -SIZE
            elif crsi_extreme_overbought:
                desired_signal = -SIZE  # Enter even against HTF on extreme
            elif rsi_overbought and hma_bear and not htf_strong_bull:
                desired_signal = -SIZE * 0.7
        
        else:  # Transition regime
            # Use both signals with reduced size
            if crsi_oversold and hma_bull:
                desired_signal = SIZE * 0.5
            elif crsi_overbought and hma_bear:
                desired_signal = -SIZE * 0.5
            elif htf_strong_bull and hma_bull:
                desired_signal = SIZE * 0.5
            elif htf_strong_bear and hma_bear:
                desired_signal = -SIZE * 0.5
        
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