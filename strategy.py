#!/usr/bin/env python3
"""
Experiment #1161: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After analyzing 848+ failed experiments, clear patterns emerge:
- Complex multi-regime logic with too many filters = 0 trades (#1150, #1155, #1157, #1159)
- Simple trend + breakout WORKS but needs regime filter to avoid choppy whipsaws
- Connors RSI proven for ETH (Sharpe +0.923 in research notes)
- Choppiness Index best meta-filter for bear/range markets (research notes)

This strategy combines PROVEN components:
1. 1d HMA(21) for MACRO trend direction (slower = fewer false signals)
2. Choppiness Index(14) for REGIME detection: >61.8 = range, <38.2 = trend
3. Connors RSI for ENTRY timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Donchian(20) breakout for MOMENTUM confirmation
5. ADX(14) > 25 for trend strength (only in trending regime)
6. 2.5x ATR(14) trailing stop (wider = fewer premature exits)

Regime-adaptive logic:
- CHOP > 61.8 (range): Mean revert with Connors RSI extremes (<15 long, >85 short)
- CHOP < 38.2 (trend): Trend follow with Donchian breakout + ADX confirmation
- CHOP 38.2-61.8 (neutral): No trades (avoid ambiguous regime)

Why this should beat Sharpe=0.612:
- Regime filter prevents trading breakouts in choppy markets (major source of losses)
- Connors RSI has 75% win rate in research for mean reversion entries
- 1d HMA provides cleaner trend signal than 12h (less noise)
- Target: 25-45 trades/year on 4h (optimal for fee drag)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 discrete (0.0, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 25-45 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_choppiness_crsi_donchian_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            for j in range(i, 0, -1):
                if close[j] > close[j-1]:
                    streak += 1
                else:
                    break
        elif close[i] < close[i-1]:
            for j in range(i, 0, -1):
                if close[j] < close[j-1]:
                    streak -= 1
                else:
                    break
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100, 50 + streak * 25)
        else:
            streak_rsi[i] = max(0, 50 + streak * 25)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures if market is trending or ranging.
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    mask = tr_smooth > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    for i in range(period * 2 - 1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_4h = calculate_adx(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx_4h[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        regime_range = choppiness[i] > 61.8  # Range-bound market
        regime_trend = choppiness[i] < 38.2  # Trending market
        regime_neutral = not regime_range and not regime_trend  # Ambiguous
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_4h[i] > 25.0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short
        
        # === BREAKOUT SIGNAL (Donchian) ===
        breakout_long = close[i] > donchian_upper[i - 1]
        breakout_short = close[i] < donchian_lower[i - 1]
        
        # === ENTRY CONDITIONS (Regime-Adaptive) ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if regime_trend and trend_strong and breakout_long and macro_bull:
            # Trend-following long: breakout in trending regime
            desired_signal = BASE_SIZE
        elif regime_range and crsi_oversold and macro_bull:
            # Mean-reversion long: oversold in range regime
            desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if regime_trend and trend_strong and breakout_short and macro_bear:
            # Trend-following short: breakout in trending regime
            desired_signal = -BASE_SIZE
        elif regime_range and crsi_overbought and macro_bear:
            # Mean-reversion short: overbought in range regime
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime becomes neutral (ambiguous)
        if in_position and regime_neutral:
            desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bull and regime not neutral
                if macro_bull and not regime_neutral:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and regime not neutral
                if macro_bear and not regime_neutral:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals