#!/usr/bin/env python3
"""
Experiment #766: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe reduces noise vs 4h while maintaining trade frequency (target 30-50/year)
2. Choppiness Index (CHOP) is superior to ADX for regime detection in crypto
   - CHOP > 61.8 = ranging (use mean reversion)
   - CHOP < 38.2 = trending (use breakout)
3. Connors RSI (CRSI) has 75% win rate for mean reversion entries
4. Donchian(20) breakout captures trend continuation with fewer false signals
5. 1d HMA(21) provides cleaner trend bias than EMA for 12h entries
6. Simple entry logic = more trades (avoid 0-trade failure mode)

Strategy design:
1. 1d HMA(21) for trend bias (aligned via mtf_data helper)
2. 12h Choppiness Index(14) for regime detection
3. 12h Connors RSI for mean reversion entries (in range regime)
4. 12h Donchian(20) for breakout entries (in trend regime)
5. 12h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from failed experiments:
- Simpler entry conditions (avoid 0 trades like #758, #760, #765)
- Choppiness instead of ADX (better for crypto range/trend detection)
- Dual regime logic (mean revert in chop, breakout in trend)
- 12h primary = fewer trades than 4h but more than 1d (sweet spot)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_donchian_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average - faster response than EMA with less lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    series = pd.Series(series)
    
    # WMA helper
    def wma(s, w):
        return s.rolling(window=w, min_periods=w).apply(
            lambda x: np.dot(x, np.arange(1, w+1)) / np.arange(1, w+1).sum(),
            raw=True
        )
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    Connors RSI Streak component.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <15 = oversold, >85 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh_ll = pd.Series(high).rolling(window=period, min_periods=period).max().values - \
            pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55  # Slightly lower threshold for more range trades
        trending_regime = chop_12h[i] < 45  # Slightly higher threshold for more trend trades
        # Neutral zone: 45-55 (hold previous position or flat)
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_12h[i] < 20  # Relaxed from 15 for more trades
        crsi_overbought = crsi_12h[i] > 80  # Relaxed from 85 for more trades
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        if ranging_regime:
            # Mean reversion long: CRSI oversold + trend not bearish
            if crsi_oversold and not trend_1d_bearish:
                desired_signal = BASE_SIZE
            
            # Mean reversion short: CRSI overbought + trend not bullish
            if crsi_overbought and not trend_1d_bullish:
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme CRSI + trend alignment
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        elif trending_regime:
            # Breakout long with trend
            if breakout_long and trend_1d_bullish:
                desired_signal = BASE_SIZE
            
            # Breakout short with trend
            if breakout_short and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            
            # Pullback entries in trend
            if trend_1d_bullish and crsi_oversold:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and crsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Only take extreme CRSI signals with trend alignment
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
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
                # Hold long if trend intact and CRSI not overbought
                if trend_1d_bullish and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_1d_bearish and crsi_12h[i] > 70:
                desired_signal = 0.0
            # Exit if breakout reverse in trending regime
            if trending_regime and close[i] < donchian_lower[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_1d_bullish and crsi_12h[i] < 30:
                desired_signal = 0.0
            # Exit if breakout reverse in trending regime
            if trending_regime and close[i] > donchian_upper[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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