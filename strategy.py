#!/usr/bin/env python3
"""
Experiment #752: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI + Dual HMA Trend

Hypothesis: After analyzing 500+ failed strategies and the success of #749/#751:
1. 12h timeframe reduces noise and fee drag vs 4h (target 20-50 trades/year)
2. Choppiness Index regime detection proven on ETH (Sharpe +0.923 in prior tests)
3. Connors RSI extremes (<15 long, >85 short) provide 75% win rate entries
4. Dual HMA trend filter (1d HMA21 + 1w HMA50) provides robust trend bias
5. Looser CRSI thresholds ( <20 / >80) ensure >=30 trades/train while maintaining quality
6. ATR(14) trailing stop 2.5x protects against adverse moves in 2022-style crashes
7. Reduced position sizing in ranging regime (0.20 vs 0.30) limits chop losses

Strategy design:
1. 1d HMA(21) for intermediate trend bias (aligned via mtf_data helper)
2. 1w HMA(50) for macro trend bias (aligned via mtf_data helper)
3. 12h Choppiness Index(14) for regime detection (trend vs range)
4. 12h Connors RSI for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
5. 12h Donchian(20) for breakout confirmation in trending regime
6. 12h ATR(14) for trailing stop and volatility scaling
7. Discrete signals: 0.0, ±0.20, ±0.30

Key improvements from #751:
- Moved from 4h to 12h primary (fewer trades, less fee drag)
- Added 1w HMA(50) as macro trend filter (better bear market protection)
- Looser CRSI thresholds to ensure trade frequency on 12h
- Simplified hold logic to reduce over-trading
- Better regime transition handling

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_dual_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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
    RSI Streak Component of Connors RSI.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    abs_streak = np.abs(streak)
    max_streak = np.nanmax(abs_streak) if np.any(~np.isnan(abs_streak)) else 1
    
    if max_streak > 0:
        streak_score = 100 * abs_streak / max_streak
    else:
        streak_score = np.zeros(n)
    
    streak_rsi = np.where(streak >= 0, streak_score, 100 - streak_score)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI.
    Measures where current close ranks vs previous N closes.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period:
        return pct_rank
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        pct_rank[i] = 100 * rank / (period - 1)
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion.
    Long: CRSI < 10-20
    Short: CRSI > 80-90
    """
    rsi_fast = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = (rsi_fast + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
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
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        
        # === TREND BIAS (Dual HMA: 1d + 1w) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend confirmation (both HTF agree)
        strong_bull = trend_1d_bullish and trend_1w_bullish
        strong_bear = trend_1d_bearish and trend_1w_bearish
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_12h[i] < 38.2
        ranging_regime = chop_12h[i] > 61.8
        
        # === CRSI SIGNALS (Connors RSI) ===
        crsi_extreme_low = crsi_12h[i] < 20
        crsi_extreme_high = crsi_12h[i] > 80
        crsi_oversold = crsi_12h[i] < 30
        crsi_overbought = crsi_12h[i] > 70
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        if trending_regime:
            # Long: Strong bullish + CRSI pullback + Donchian support
            if strong_bull and crsi_oversold and close[i] > donch_lower[i-1]:
                desired_signal = BASE_SIZE
            
            # Short: Strong bearish + CRSI rally + Donchian resistance
            if strong_bear and crsi_overbought and close[i] < donch_upper[i-1]:
                desired_signal = -BASE_SIZE
            
            # Trend continuation with SMA confirmation
            if strong_bull and above_sma50 and above_sma200 and crsi_12h[i] > 35 and crsi_12h[i] < 65:
                desired_signal = BASE_SIZE
            
            if strong_bear and below_sma50 and below_sma200 and crsi_12h[i] > 35 and crsi_12h[i] < 65:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion long: CRSI extreme low + no strong bearish trend
            if crsi_extreme_low and not strong_bear:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: CRSI extreme high + no strong bullish trend
            if crsi_extreme_high and not strong_bull:
                desired_signal = -REDUCED_SIZE
            
            # Pure mean reversion at range boundaries
            if crsi_extreme_low and close[i] < donch_lower[i-1] * 1.02:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_high and close[i] > donch_upper[i-1] * 0.98:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on CRSI extremes + strong trend alignment
            if crsi_extreme_low and strong_bull and above_sma50:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_high and strong_bear and below_sma50:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if strong_bull and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                if strong_bear and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if strong_bear and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if strong_bull and crsi_12h[i] < 30:
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