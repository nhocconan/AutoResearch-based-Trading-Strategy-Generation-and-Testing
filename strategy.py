#!/usr/bin/env python3
"""
Experiment #1012: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + KAMA + CRSI

Hypothesis: After 735 failed strategies, the key insight is that SINGLE regime strategies
fail because crypto alternates between trending and ranging. This strategy uses:

1. CHOPPINESS INDEX (CHOP) for regime detection:
   - CHOP > 61.8 = ranging market → use mean reversion (CRSI extremes)
   - CHOP < 38.2 = trending market → use trend following (KAMA crossover)
   - Between = hold existing positions, no new entries

2. KAMA (Kaufman Adaptive Moving Average): Adapts to volatility
   - More responsive in trends, smoother in chop
   - Better than HMA/EMA for regime-switching strategies

3. Connors RSI for mean reversion entries in choppy regimes:
   - Long: CRSI < 20 + price > 1d HMA21
   - Short: CRSI > 80 + price < 1d HMA21

4. KAMA Crossover for trend entries in trending regimes:
   - Long: KAMA(10) > KAMA(40) + price > 1d HMA21
   - Short: KAMA(10) < KAMA(40) + price < 1d HMA21

5. 1d HMA21: Single HTF filter for macro trend bias
   - Only take longs when price > 1d HMA (bullish macro)
   - Only take shorts when price < 1d HMA (bearish macro)

6. ATR Trailing Stop: 2.5x ATR for risk management

Why 12h works:
- Target 20-50 trades/year (vs 100+ on 4h)
- Less noise, cleaner signals
- Fee drag minimized (0.05% per trade × 40 trades = 2% annual drag)

Critical fixes from failed experiments:
- DUAL regime (not single) — adapts to market conditions
- SINGLE HTF (1d HMA) — not multiple conflicting HTF filters
- RELAXED thresholds (CRSI 20/80 not 10/90, CHOP 38.2/61.8 standard)
- KAMA instead of HMA for primary trend (more adaptive)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_kama_crsi_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * (streak_abs[i] / (streak_abs[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_lower = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility via Efficiency Ratio
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    kama_fast_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=40)
    
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
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(kama_fast_12h[i]) or np.isnan(kama_slow_12h[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_12h[i] > 61.8  # Ranging market
        regime_trend = chop_12h[i] < 38.2  # Trending market
        regime_neutral = not regime_chop and not regime_trend  # Transition zone
        
        # === CRSI SIGNALS (Mean Reversion for Choppy Regime) ===
        crsi_extreme_oversold = crsi_12h[i] < 20
        crsi_extreme_overbought = crsi_12h[i] > 80
        crsi_oversold = crsi_12h[i] < 30
        crsi_overbought = crsi_12h[i] > 70
        
        # === KAMA CROSSOVER (Trend Following for Trending Regime) ===
        kama_bullish = kama_fast_12h[i] > kama_slow_12h[i]
        kama_bearish = kama_fast_12h[i] < kama_slow_12h[i]
        kama_cross_long = kama_bullish and kama_fast_12h[i-1] <= kama_slow_12h[i-1]
        kama_cross_short = kama_bearish and kama_fast_12h[i-1] >= kama_slow_12h[i-1]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop and macro_bull:
            # Mean reversion in choppy bullish market
            if crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            elif crsi_oversold:
                desired_signal = REDUCED_SIZE
        elif regime_trend and macro_bull:
            # Trend following in trending bullish market
            if kama_cross_long:
                desired_signal = BASE_SIZE
            elif kama_bullish and crsi_oversold:
                desired_signal = REDUCED_SIZE
        elif regime_neutral and macro_bull:
            # Hold through transition if already long, relaxed entry
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop and macro_bear:
            # Mean reversion in choppy bearish market
            if crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_overbought:
                desired_signal = -REDUCED_SIZE
        elif regime_trend and macro_bear:
            # Trend following in trending bearish market
            if kama_cross_short:
                desired_signal = -BASE_SIZE
            elif kama_bearish and crsi_overbought:
                desired_signal = -REDUCED_SIZE
        elif regime_neutral and macro_bear:
            # Hold through transition if already short, relaxed entry
            if crsi_extreme_overbought:
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
                # Hold long if macro bullish and CRSI not extreme overbought
                if macro_bull and crsi_12h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not extreme oversold
                if macro_bear and crsi_12h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + CRSI overbought
            if macro_bear and crsi_12h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + CRSI oversold
            if macro_bull and crsi_12h[i] < 35:
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