#!/usr/bin/env python3
"""
Experiment #1516: 12h Primary + 1d HTF — KAMA Adaptive Trend + Connors RSI + ADX Regime

Hypothesis: Based on #1506 success (12h HMA+RSI Sharpe=0.138) and ETH Connors RSI success (Sharpe=0.923),
combining KAMA (adaptive to volatility) with Connors RSI (proven mean reversion) and ADX regime filter
should beat the current best (Sharpe=0.618). Key insights from 1100+ failed strategies:

1. KAMA adapts better than HMA/EMA in choppy markets (2022 crash, 2025 bear)
2. Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 has 75% win rate in backtests
3. ADX regime filter prevents mean reversion in strong trends (and vice versa)
4. 12h timeframe naturally generates 20-50 trades/year (fee efficient)
5. MUST ensure trades happen: loose entry bands, multiple entry paths

Design:
- 1d HMA(21) for macro trend bias (HTF filter)
- 12h KAMA(14, ER=10) for adaptive primary trend
- Connors RSI(3,2,100) for mean reversion entries
- ADX(14) for regime detection (>25=trend, <20=range)
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.25-0.30 (discrete levels)
- Target: 30-60 trades/train (4 years), 8-15 trades/test (15 months)

Timeframe: 12h (as required by experiment)
HTF: 1d (daily trend bias)
Position Size: 0.25-0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades >= 30 train / >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_adx_1d_regime_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_kama(close, period=14, er_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility via Efficiency Ratio
    KAMA[i] = KAMA[i-1] + SC * (price - KAMA[i-1])
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Sum of absolute changes over er_period
    """
    n = len(close)
    if n < period + er_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        if change == 0:
            er[i] = 0
        else:
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 0:
                er[i] = change / noise
    
    # Smoothing Constant
    fast_sc = 2.0 / (2.0 + 1.0)
    slow_sc = 2.0 / (2.0 + 30.0)
    
    sc = np.full(n, np.nan)
    mask = ~np.isnan(er)
    sc[mask] = (er[mask] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = streak[i - 1]
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank of returns
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if len(window) == rank_period:
            pct_rank[i] = np.sum(window[:-1] < returns[i]) / (rank_period - 1) * 100.0
    
    # Combine
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + pct_rank[mask]) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    adx = np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    # DX
    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * di_diff[mask2] / di_sum[mask2]
    
    # ADX (smoothed DX)
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2:] = adx_raw[period * 2:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, period=14, er_period=10)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Appropriate size for 12h (20-50 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(kama[i]) or np.isnan(adx[i]):
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
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA) - adaptive confirmation ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === ADX REGIME ===
        trending = adx[i] > 25.0
        ranging = adx[i] < 20.0
        
        # === CONNORS RSI - Mean Reversion Signals ===
        # Long: CRSI < 20 (oversold)
        crsi_oversold = crsi[i] < 20.0
        # Short: CRSI > 80 (overbought)
        crsi_overbought = crsi[i] > 80.0
        
        # === LOOSE ENTRY CONDITIONS (ensure trades happen) ===
        # Multiple paths to entry to avoid 0 trades
        
        desired_signal = 0.0
        
        # LONG ENTRIES (multiple paths)
        # Path 1: Trending + Daily Bull + KAMA Bull + CRSI pullback (not oversold)
        if trending and daily_bull and kama_bull and crsi[i] < 50.0:
            desired_signal = BASE_SIZE
        # Path 2: Ranging + Daily Bull + CRSI oversold (mean reversion)
        elif ranging and daily_bull and crsi_oversold:
            desired_signal = BASE_SIZE * 0.9
        # Path 3: Daily Bull + KAMA Bull + CRSI < 40 (simple trend pullback)
        elif daily_bull and kama_bull and crsi[i] < 40.0:
            desired_signal = BASE_SIZE * 0.8
        # Path 4: Daily Bull + KAMA Bull (fallback for trades)
        elif daily_bull and kama_bull:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT ENTRIES (multiple paths)
        # Path 1: Trending + Daily Bear + KAMA Bear + CRSI rally (not overbought)
        elif trending and daily_bear and kama_bear and crsi[i] > 50.0:
            desired_signal = -BASE_SIZE
        # Path 2: Ranging + Daily Bear + CRSI overbought (mean reversion)
        elif ranging and daily_bear and crsi_overbought:
            desired_signal = -BASE_SIZE * 0.9
        # Path 3: Daily Bear + KAMA Bear + CRSI > 60 (simple trend pullback)
        elif daily_bear and kama_bear and crsi[i] > 60.0:
            desired_signal = -BASE_SIZE * 0.8
        # Path 4: Daily Bear + KAMA Bear (fallback for trades)
        elif daily_bear and kama_bear:
            desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.70:
            final_signal = BASE_SIZE * 0.85
        elif desired_signal >= BASE_SIZE * 0.50:
            final_signal = BASE_SIZE * 0.70
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.70:
            final_signal = -BASE_SIZE * 0.85
        elif desired_signal <= -BASE_SIZE * 0.50:
            final_signal = -BASE_SIZE * 0.70
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