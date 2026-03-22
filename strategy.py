#!/usr/bin/env python3
"""
Experiment #447: 1d Primary + 1w HTF — Simplified Connors RSI + Choppiness Regime

Hypothesis: After analyzing 446 experiments, clear pattern emerges:
1. 1d timeframe has best risk/reward (current best Sharpe=0.435 is 1d-based)
2. Connors RSI with RELAXED thresholds (25/75 not 10/90) generates enough trades
3. Choppiness Index regime filter works BUT must not be too restrictive
4. 1w HMA provides major trend bias without killing trade frequency
5. SIMPLER logic = more trades = better statistical significance (see exp 440, 445 = 0 trades)

Key lessons from failures:
- Exp 440, 445: Sharpe=0.000 because 0 trades (filters too strict)
- Exp 435-444: All negative Sharpe from over-filtering
- Exp 446: Positive Sharpe (0.134) with relaxed thresholds

Why this might beat current best (Sharpe=0.435):
- Connors RSI proven 75% win rate in academic research
- Choppiness regime adapts to market conditions (trend vs range)
- 1d TF has minimal fee drag (~10-30 trades/year target)
- ATR 3.0x trailing stop protects in 2022-style crashes
- Asymmetric sizing: 0.30 long, 0.25 short (bear market protection)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 3.0 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_1w_regime_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate in research. RELAXED thresholds for more trades.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    # Combine components
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, period=21)
    hma_1d_50 = calculate_hma(close, period=50)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull bias (favor longs)
        # Price below 1w HMA = bear bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION (RELAXED for more trades) ===
        # CHOP > 55 = ranging (mean reversion)
        # CHOP < 45 = trending (trend follow)
        is_ranging = chop_1d[i] > 50.0
        is_trending = chop_1d[i] < 50.0
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_21[i] > hma_1d_50[i]
        hma_bearish = hma_1d_21[i] < hma_1d_50[i]
        
        # === CONNORS RSI SIGNALS (RELAXED thresholds for more trades) ===
        # Original: <10/>90 (too strict, 0 trades)
        # Relaxed: <25/>75 (proven in exp 446)
        crsi_oversold = crsi_1d[i] < 30.0
        crsi_overbought = crsi_1d[i] > 70.0
        crsi_extreme_oversold = crsi_1d[i] < 15.0
        crsi_extreme_overbought = crsi_1d[i] > 85.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01  # within 1% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.99  # within 1% of upper band
        
        # === ENTRY LOGIC — REGIME ADAPTIVE (SIMPLIFIED FOR MORE TRADES) ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence paths for more trades)
        if bull_regime or above_sma200:
            # Path 1: Ranging market + CRSI oversold (mean reversion)
            if is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # Path 2: Trending market + HMA bullish + CRSI pullback
            elif is_trending and hma_bullish and crsi_1d[i] < 45.0:
                new_signal = LONG_SIZE
            # Path 3: CRSI extreme oversold (works in any regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_SIZE
            # Path 4: HMA crossover + CRSI confirmation
            elif hma_bullish and crsi_1d[i] < 40.0:
                new_signal = LONG_SIZE * 0.9
            # Path 5: Bollinger lower band + CRSI oversold
            elif near_bb_lower and crsi_oversold:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES (multiple confluence paths for more trades)
        if bear_regime or below_sma200:
            # Path 1: Ranging market + CRSI overbought (mean reversion)
            if is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: Trending market + HMA bearish + CRSI bounce
            elif is_trending and hma_bearish and crsi_1d[i] > 55.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 3: CRSI extreme overbought (works in any regime)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 4: HMA crossover + CRSI confirmation
            elif hma_bearish and crsi_1d[i] > 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
            # Path 5: Bollinger upper band + CRSI overbought
            elif near_bb_upper and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: CRSI < 35 + HMA bullish (simpler entry)
            if bull_regime and hma_bullish and crsi_1d[i] < 35.0:
                new_signal = LONG_SIZE * 0.6
            # Short: CRSI > 65 + HMA bearish (simpler entry)
            elif bear_regime and hma_bearish and crsi_1d[i] > 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_1d[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1d[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals