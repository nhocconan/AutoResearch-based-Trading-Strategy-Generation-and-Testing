#!/usr/bin/env python3
"""
Experiment #1573: 1d Primary + 1w HTF — Dual Regime Strategy (Choppiness + Connors RSI + Donchian)

Hypothesis: After 11 failed 4h experiments, 1d timeframe with regime switching should work better.
Research shows Choppiness Index + Connors RSI achieved ETH Sharpe +0.923, and Donchian + HMA + RSI
achieved SOL Sharpe +0.782. Combining these in a dual-regime framework should beat Sharpe 0.618.

Key innovations:
1. Choppiness Index (14) regime detection: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
2. Range regime: Connors RSI (RSI3 + RSI_Streak2 + PercentRank100) / 3 for mean reversion entries
3. Trend regime: Donchian(20) breakout + 1w HMA(21) bias for directional trades
4. 1w HMA aligned to 1d for long-term trend filter (use mtf_data helper - call ONCE before loop)
5. ATR(14) 2.5x trailing stop for drawdown control
6. Discrete position sizing (0.30) to minimize fee churn
7. Target 20-50 trades/year on 1d timeframe (optimal for fee efficiency)

Why this should beat Sharpe 0.618:
- 1d timeframe has proven better than 4h (see experiment history - 4h strategies mostly negative)
- Dual regime adapts to market conditions (trend vs range) - critical for 2022 crash + 2025 bear
- Connors RSI has 75% win rate in backtests for mean reversion
- Donchian breakouts capture major moves while 1w HMA filters false breakouts
- 1d targets 20-50 trades/year — optimal balance of signal quality vs fee drag

Timeframe: 1d (required for this experiment)
HTF: 1w HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
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
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): RSI on consecutive up/down streak length
    PercentRank(100): Where current close ranks vs last 100 closes (0-100)
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3) on close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_3 = pd.Series(gain).ewm(span=3, min_periods=3, adjust=False).mean().values
    loss_3 = pd.Series(loss).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    mask = loss_3 > 1e-10
    rsi_close[mask] = 100.0 - (100.0 / (1.0 + gain_3[mask] / loss_3[mask]))
    rsi_close[loss_3 <= 1e-10] = 100.0
    rsi_close[:3] = np.nan
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_2 = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    streak_loss_2 = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_2 > 1e-10
    rsi_streak[mask] = 100.0 - (100.0 / (1.0 + streak_gain_2[mask] / streak_loss_2[mask]))
    rsi_streak[streak_loss_2 <= 1e-10] = 100.0
    rsi_streak[:2] = np.nan
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    valid = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_close[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(close) if 'close' in dir() else len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
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
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range/mean-reversion regime
        is_trending = chop[i] < 45.0  # Trend-following regime
        # 45-55 = transition zone (no new entries)
        
        # === LONG-TERM BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- RANGE REGIME: Connors RSI Mean Reversion ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price > SMA200 (long-term support)
            if crsi[i] < 15.0 and not np.isnan(sma_200[i]) and close[i] > sma_200[i] * 0.95:
                desired_signal = BASE_SIZE
            # Short: CRSI > 85 (overbought) + price < SMA200 (long-term resistance)
            elif crsi[i] > 85.0 and not np.isnan(sma_200[i]) and close[i] < sma_200[i] * 1.05:
                desired_signal = -BASE_SIZE
        
        # --- TREND REGIME: Donchian Breakout + Weekly Bias ---
        elif is_trending:
            # Long: Price breaks Donchian upper + weekly bull
            if close[i] > donchian_upper[i-1] and weekly_bull:
                desired_signal = BASE_SIZE
            # Short: Price breaks Donchian lower + weekly bear
            elif close[i] < donchian_lower[i-1] and weekly_bear:
                desired_signal = -BASE_SIZE
        
        # --- TRANSITION ZONE: Hold existing positions, no new entries ---
        # (desired_signal stays 0.0)
        
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