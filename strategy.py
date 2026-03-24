#!/usr/bin/env python3
"""
Experiment #1622: 12h Primary + 1d/1w HTF — Dual Regime Strategy (Chop/Trend)

Hypothesis: 12h timeframe with regime-switching logic will outperform 4h strategies.
Key insight from research: Choppiness Index + Connors RSI achieved ETH Sharpe +0.923,
and Donchian + HMA + RSI achieved SOL Sharpe +0.782. Combining these with HTF bias
should work across all symbols (BTC/ETH/SOL) in both bull and bear markets.

Strategy Logic:
1. REGIME DETECTION: Choppiness Index determines market state
   - CHOP > 55 = Choppy (use mean reversion with CRSI)
   - CHOP < 45 = Trending (use trend following with KAMA + Donchian)
   
2. HTF BIAS: 1d + 1w HMA alignment prevents counter-trend trades
   - Both daily and weekly must agree for strong bias
   - At least daily must support for any entry
   
3. ENTRY CONDITIONS (loose enough for 20-50 trades/year):
   - Chop regime: CRSI < 15 (long) or > 85 (short) + HTF support
   - Trend regime: KAMA direction + Donchian breakout OR CRSI pullback
   
4. RISK: 2.2x ATR trailing stop, position size 0.28 (28% of capital)

Why this should beat Sharpe 0.618:
- 12h TF = fewer trades, less fee drag (target 25-50 trades/year)
- Dual regime adapts to 2022 crash (chop) and 2021 bull (trend)
- HTF bias (1d + 1w) prevents major counter-trend losses
- Looser thresholds ensure >= 10 trades/symbol on train
- Works on BTC/ETH (mean reversion) and SOL (trend)

Timeframe: 12h (required for exp #1622)
HTF: 1d HMA + 1w HMA (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    adx = np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smoothed DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        sum_plus_dm = np.sum(plus_dm[i-period+1:i+1])
        sum_minus_dm = np.sum(minus_dm[i-period+1:i+1])
        sum_tr = np.sum(tr[i-period+1:i+1])
        
        if sum_tr > 1e-10:
            plus_di[i] = 100.0 * sum_plus_dm / sum_tr
            minus_di[i] = 100.0 * sum_minus_dm / sum_tr
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = SMA of DX
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i-period+1:i+1])
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - CHOP > 61.8 = range, CHOP < 38.2 = trend"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - composite mean reversion indicator"""
    n = len(close)
    if n < rank_period + 10:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(close, 3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_price = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_price[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_price[loss_smooth <= 1e-10] = 100.0
    rsi_price[:rsi_period] = np.nan
    
    # RSI(streak, 2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask2 = streak_loss_smooth > 1e-10
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask2] / streak_loss_smooth[mask2]))
    rsi_streak[streak_loss_smooth <= 1e-10] = 100.0
    rsi_streak[:streak_period] = np.nan
    
    # PercentRank(close, 100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < close[i]) / len(window) * 100.0
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_price) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_price[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother than EMA with less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
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
    """Donchian Channels for breakout detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx = calculate_adx(high, low, close, period=14)
    
    # KAMA for adaptive trend
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Donchian channels for breakouts
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(chop[i]) or
            np.isnan(crsi[i]) or np.isnan(adx[i]) or np.isnan(kama[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0  # Range market (slightly lower threshold for more trades)
        is_trending = chop[i] < 45.0  # Trend market
        
        # === TREND BIAS (1d + 1w HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        strong_bull = daily_bull and weekly_bull
        strong_bear = daily_bear and weekly_bear
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 22.0  # Slightly lower for more trades
        adx_weak = adx[i] < 18.0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 15.0  # LOOSE for more trades
        crsi_overbought = crsi[i] > 85.0  # LOOSE for more trades
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY MARKET - Mean Reversion with CRSI
        if is_choppy:
            if crsi_oversold and (daily_bull or not strong_bear):
                desired_signal = BASE_SIZE
            elif crsi_overbought and (daily_bear or not strong_bull):
                desired_signal = -BASE_SIZE
        
        # REGIME 2: TRENDING MARKET - Trend Following
        elif is_trending and adx_strong:
            # Trend following with KAMA + HTF bias + Donchian breakout
            if kama_bull and (strong_bull or daily_bull) and donchian_breakout_long:
                desired_signal = BASE_SIZE
            elif kama_bear and (strong_bear or daily_bear) and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            # Also enter on pullback in trend
            elif kama_bull and strong_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            elif kama_bear and strong_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
        
        # REGIME 3: NEUTRAL/TRANSITION - Hold existing or exit
        else:
            if in_position:
                desired_signal = BASE_SIZE if position_side > 0 else -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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