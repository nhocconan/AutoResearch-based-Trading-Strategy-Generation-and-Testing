#!/usr/bin/env python3
"""
Experiment #772: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe reduces fee drag significantly vs 4h/1h (target 20-50 trades/year)
2. Choppiness Index (CHOP) is superior regime filter for crypto mean reversion
3. Connors RSI (CRSI) has 75% win rate for oversold/overbought reversals
4. 1d HMA(21) + 1w HMA(21) provide cleaner trend bias than EMA (less lag)
5. Simpler entry logic = more trades (avoid 0-trade failure mode)
6. ATR trailing stop (2.5x) protects against 2022-style crashes

Strategy design:
1. 12h Choppiness Index (14) for regime: CHOP>61.8=ranging, CHOP<38.2=trending
2. 12h Connors RSI for entries: <15=oversold long, >85=overbought short
3. 1d HMA(21) for intermediate trend bias (aligned via mtf_data)
4. 1w HMA(21) for major trend filter (aligned via mtf_data)
5. 12h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.20, ±0.30
7. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key improvements from #764:
- Higher timeframe (12h vs 4h) = fewer trades, less fee drag
- Choppiness Index regime (proven for ETH mean reversion)
- HMA instead of EMA (faster response, less lag)
- Dual HTF (1d + 1w) for stronger trend confirmation
- Simpler entry conditions to ensure trade generation

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    s = pd.Series(series)
    half = s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    full = s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = (2 * half - full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hull.values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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
    """Connors RSI Streak component."""
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
    """Connors RSI Percent Rank component."""
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
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market ranging vs trending.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === TREND BIAS (HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend when both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        ranging_regime = chop_12h[i] > 61.8
        trending_regime = chop_12h[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        crsi_neutral_low = 30 < crsi_12h[i] < 45
        crsi_neutral_high = 55 < crsi_12h[i] < 70
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) - Mean Reversion ===
        if ranging_regime:
            # Long: CRSI oversold + not strong bearish trend
            if crsi_oversold and not strong_bearish:
                desired_signal = BASE_SIZE if crsi_extreme_oversold else REDUCED_SIZE
            
            # Short: CRSI overbought + not strong bullish trend
            if crsi_overbought and not strong_bullish:
                desired_signal = -BASE_SIZE if crsi_extreme_overbought else -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) - Trend Follow ===
        elif trending_regime:
            # Long pullback: strong bullish + CRSI neutral low
            if strong_bullish and crsi_neutral_low:
                desired_signal = BASE_SIZE
            
            # Short pullback: strong bearish + CRSI neutral high
            if strong_bearish and crsi_neutral_high:
                desired_signal = -BASE_SIZE
            
            # Breakout continuation
            if strong_bullish and crsi_12h[i] > 50 and crsi_12h[i] < 70:
                desired_signal = REDUCED_SIZE
            
            if strong_bearish and crsi_12h[i] < 50 and crsi_12h[i] > 30:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME LOGIC (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only extreme CRSI + trend alignment
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
                # Hold long if 1d trend intact and CRSI not overbought
                if trend_1d_bullish and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE if strong_bullish else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE if strong_bearish else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses or CRSI very overbought
            if trend_1d_bearish and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses or CRSI very oversold
            if trend_1d_bullish and crsi_12h[i] < 30:
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