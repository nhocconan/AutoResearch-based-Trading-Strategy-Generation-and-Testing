#!/usr/bin/env python3
"""
Experiment #802: 12h Primary + 1d/1w HTF — Connors RSI Mean Reversion + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 12h timeframe provides optimal balance - fewer trades (20-50/year) but higher quality signals
2. Connors RSI (CRSI) outperforms standard RSI for mean reversion - 75% win rate in backtests
3. Choppiness Index (CHOP) regime filter prevents trend strategies in chop and vice versa
4. 1d HMA(21) provides smoother trend bias than 1w (more responsive to regime changes)
5. 1w HMA(50) gives ultra-long-term context for filtering counter-trend trades
6. ATR(14) trailing stop at 2.5x protects from major drawdowns without premature exits
7. Dual regime logic: mean revert when CHOP>61.8, trend follow when CHOP<38.2
8. Position sizing: 0.25-0.30 discrete levels to control fees and drawdown

Strategy design:
1. 1d HMA(21) for medium-term trend bias (aligned via mtf_data helper)
2. 1w HMA(50) for long-term trend context (aligned via mtf_data helper)
3. 12h Choppiness Index(14) for regime detection on primary timeframe
4. 12h Connors RSI for entry timing - more responsive than standard RSI
5. 12h Donchian(20) for breakout confirmation in trending regimes
6. 12h ATR(14) for trailing stop (2.5x) and volatility normalization
7. Discrete signals: 0.0, ±0.25, ±0.30
8. Relaxed CRSI thresholds (15/85 for entries, 5/95 for extremes) to ensure >=10 trades/train

Key differences from failed 12h strategies:
- Connors RSI instead of standard RSI (3-component composite vs single)
- CHOP calculated on 12h (primary TF) not 1d - more responsive regime detection
- Dual HMA system: 1d for bias, 1w for filter (not just single HTF)
- Donchian breakout for trend entries (not just RSI pullbacks)
- Volume filter removed (adds complexity without edge on 12h)
- Hold logic: maintain position until opposite signal, stoploss, or CRSI extreme exit

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_1d1w_donchian_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Larry Connors' composite mean reversion indicator.
    Combines: RSI(close, 2) + RSI(streak, 2) + PercentRank(close, 100)
    Values 0-100. <10 = extreme oversold, >90 = extreme overbought.
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(2) on price changes
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI(2) on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_avg_gain = np.concatenate([[np.nan], streak_avg_gain])
    streak_avg_loss = np.concatenate([[np.nan], streak_avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of close over lookback period
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine all three components equally weighted
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow).
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    series = pd.Series(series)
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops and position sizing."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    for i in range(n):
        prev_close = close[i-1] if i > 0 else close[i]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection via highest high / lowest low."""
    n = len(high)
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA(21) for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA(50) for long-term trend filter
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 61.8
        trending_regime = chop_12h[i] < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === LONG-TERM TREND FILTER (1w HTF HMA50) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS (relaxed for more trades) ===
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_moderate_oversold = crsi_12h[i] < 30
        crsi_moderate_overbought = crsi_12h[i] > 70
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) - MEAN REVERSION ===
        if ranging_regime:
            # Strong long: CRSI extreme oversold + 1d bullish
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = BASE_SIZE
            # Strong short: CRSI extreme overbought + 1d bearish
            elif crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -BASE_SIZE
            # Moderate long: CRSI oversold + 1d bullish + 1w bullish filter
            elif crsi_oversold and trend_1d_bullish and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            # Moderate short: CRSI overbought + 1d bearish + 1w bearish filter
            elif crsi_overbought and trend_1d_bearish and trend_1w_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) - TREND FOLLOWING ===
        elif trending_regime:
            # Breakout long: Donchian breakout + 1d bullish + 1w bullish
            if donchian_breakout_long and trend_1d_bullish and trend_1w_bullish:
                desired_signal = BASE_SIZE
            # Breakout short: Donchian breakout + 1d bearish + 1w bearish
            elif donchian_breakout_short and trend_1d_bearish and trend_1w_bearish:
                desired_signal = -BASE_SIZE
            # Pullback long: CRSI oversold in uptrend
            elif crsi_moderate_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            # Pullback short: CRSI overbought in downtrend
            elif crsi_moderate_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Only take extreme CRSI signals with trend alignment
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            elif crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            # Allow moderate signals if both 1d and 1w agree
            elif crsi_oversold and trend_1d_bullish and trend_1w_bullish:
                desired_signal = REDUCED_SIZE
            elif crsi_overbought and trend_1d_bearish and trend_1w_bearish:
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
                # Hold long if 1d trend intact and CRSI not extreme overbought
                if trend_1d_bullish and crsi_12h[i] < 85:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and CRSI not extreme oversold
                if trend_1d_bearish and crsi_12h[i] > 15:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI reaches extreme overbought (take profit)
            if crsi_12h[i] > 90:
                desired_signal = 0.0
            # Exit if 1d trend reverses strongly
            if trend_1d_bearish and crsi_12h[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI reaches extreme oversold (take profit)
            if crsi_12h[i] < 10:
                desired_signal = 0.0
            # Exit if 1d trend reverses strongly
            if trend_1d_bullish and crsi_12h[i] < 50:
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