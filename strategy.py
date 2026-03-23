#!/usr/bin/env python3
"""
Experiment #753: 1d Primary + 1w HTF — Dual Regime with Connors RSI + Choppiness + HMA

Hypothesis: After analyzing 500+ failed strategies and the success of #751 (Sharpe=0.342):
1. 1d primary timeframe naturally produces fewer trades (20-50/year target) - less fee drag
2. 1w HMA(21) provides smoother trend bias than 1d, reducing whipsaw in bear markets
3. Connors RSI with LOOSER thresholds (<20/>80 instead of <15/>85) ensures >=30 trades/train
4. Choppiness Index regime switch allows adaptive behavior: trend-follow in trends, mean-revert in ranges
5. Donchian(20) breakout confirmation adds confluence for trending regime entries
6. ATR(14) 2.5x trailing stop protects against adverse moves without premature exits
7. Discrete signal levels (0.0, ±0.25, ±0.30) minimize fee churn from signal changes

Key improvements from #751:
- Primary timeframe 1d instead of 4h (fewer trades, less noise)
- HTF 1w instead of 1d (smoother trend filter)
- Looser CRSI thresholds to ensure trade frequency (critical for 1d)
- Simplified regime logic (2 regimes: trending vs ranging)
- Better hold logic to maintain positions through favorable trends

Strategy design:
1. 1w HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for regime detection
3. 1d Connors RSI for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. 1d Donchian(20) for breakout confirmation in trending regime
5. 1d ATR(14) for trailing stop
6. Discrete signals: 0.0, ±0.25, ±0.30

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_hma_1w_donchian_atr_v1"
timeframe = "1d"
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
    Measures consecutive up/down days.
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
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    max_streak = np.max(abs_streak[~np.isnan(abs_streak)]) if np.any(~np.isnan(abs_streak)) else 1
    
    if max_streak > 0:
        streak_score = 100 * abs_streak / max_streak
    else:
        streak_score = np.zeros(n)
    
    # Apply direction (up streak = bullish, down streak = bearish)
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
    Long: CRSI < 15-20
    Short: CRSI > 80-85
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(chop_1d[i]):
            continue
        
        # === TREND BIAS (1w HTF HMA) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_1d[i] < 38.2
        ranging_regime = chop_1d[i] > 61.8
        
        # === CRSI SIGNALS (Connors RSI - LOOSER thresholds for trade frequency) ===
        crsi_extreme_low = crsi_1d[i] < 20
        crsi_extreme_high = crsi_1d[i] > 80
        crsi_oversold = crsi_1d[i] < 35
        crsi_overbought = crsi_1d[i] > 65
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donch_upper[i-1]
        breakout_short = close[i] < donch_lower[i-1]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        if trending_regime:
            # Long: 1w bullish + CRSI pullback + Donchian support or breakout
            if trend_1w_bullish and crsi_oversold:
                if close[i] > donch_lower[i-1] or breakout_long:
                    desired_signal = BASE_SIZE
            
            # Short: 1w bearish + CRSI rally + Donchian resistance or breakdown
            if trend_1w_bearish and crsi_overbought:
                if close[i] < donch_upper[i-1] or breakout_short:
                    desired_signal = -BASE_SIZE
            
            # Strong trend continuation (breakout with trend)
            if trend_1w_bullish and above_sma50 and breakout_long:
                desired_signal = BASE_SIZE
            
            if trend_1w_bearish and below_sma50 and breakout_short:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion long: CRSI extreme low + 1w bullish or neutral bias
            if crsi_extreme_low and (trend_1w_bullish or not trend_1w_bearish):
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: CRSI extreme high + 1w bearish or neutral bias
            if crsi_extreme_high and (trend_1w_bearish or not trend_1w_bullish):
                desired_signal = -REDUCED_SIZE
            
            # Donchian mean reversion in range
            if crsi_extreme_low and close[i] < donch_lower[i-1]:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_high and close[i] > donch_upper[i-1]:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on CRSI extremes + trend alignment
            if crsi_extreme_low and trend_1w_bullish and above_sma50:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_high and trend_1w_bearish and below_sma50:
                desired_signal = -REDUCED_SIZE
            
            # Breakout in neutral regime with trend confirmation
            if trend_1w_bullish and breakout_long and above_sma50:
                desired_signal = BASE_SIZE
            
            if trend_1w_bearish and breakout_short and below_sma50:
                desired_signal = -BASE_SIZE
        
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
                if trend_1w_bullish and crsi_1d[i] < 70:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                if trend_1w_bearish and crsi_1d[i] > 30:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if trend_1w_bearish and crsi_1d[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_1w_bullish and crsi_1d[i] < 35:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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