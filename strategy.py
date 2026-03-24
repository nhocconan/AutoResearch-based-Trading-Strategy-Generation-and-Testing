#!/usr/bin/env python3
"""
Experiment #946: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Daily timeframe with weekly HTF bias provides optimal trade frequency
(20-50 trades/year) while avoiding fee drag. Choppiness Index detects regime
(range vs trend), allowing dual strategy: Connors RSI mean reversion in chop,
HMA trend following in trends. Weekly HMA provides directional bias filter.

Key innovations:
1. Choppiness Index (14) regime detection: >61.8=range, <38.2=trend
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Weekly HMA(21) for HTF directional bias
4. Dual entry logic based on regime (mean revert vs trend follow)
5. LOOSE entry thresholds to ensure ≥10 trades/train, ≥3/test
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (LOOSE to guarantee trades):
- RANGE regime (CHOP>55): CRSI<15 long, CRSI>85 short (loose mean reversion)
- TREND regime (CHOP<45): HMA crossover + HTF bias alignment
- RSI filter only avoids extremes (RSI>80 no long, RSI<20 no short)

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_weekly_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is chopping/ranging or trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <10 = oversold (long), >90 = overbought (short)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n, dtype=np.int32)
    
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            # Longer streak = more extreme
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 25.0
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_range_regime = chop_value > 55.0  # Loose threshold for more trades
        is_trend_regime = chop_value < 45.0  # Loose threshold for more trades
        # Between 45-55 = neutral, can use either strategy
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_1d_16[i-1]) and not np.isnan(hma_1d_48[i-1]):
            hma_crossover_long = (hma_1d_16[i-1] <= hma_1d_48[i-1]) and (hma_1d_16[i] > hma_1d_48[i])
            hma_crossover_short = (hma_1d_16[i-1] >= hma_1d_48[i-1]) and (hma_1d_16[i] < hma_1d_48[i])
        
        # === 1d HMA TREND ===
        hma_1d_bull = hma_1d_16[i] > hma_1d_48[i]
        hma_1d_bear = hma_1d_16[i] < hma_1d_48[i]
        
        # === RSI FILTER (VERY LOOSE) ===
        rsi_overbought = rsi_14[i] > 80.0  # Very loose
        rsi_oversold = rsi_14[i] < 20.0   # Very loose
        
        # === CRSI EXTREMES (LOOSE) ===
        crsi_oversold = crsi[i] < 20.0  # Loose for more long signals
        crsi_overbought = crsi[i] > 80.0  # Loose for more short signals
        
        # === ENTRY LOGIC (DUAL REGIME - LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion with Connors RSI
        if is_range_regime:
            # Long on CRSI oversold (loose threshold)
            if crsi_oversold and not rsi_overbought:
                desired_signal = SIZE_BASE
            # Short on CRSI overbought (loose threshold)
            elif crsi_overbought and not rsi_oversold:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Trend following with HMA + HTF bias
        elif is_trend_regime:
            # Long: HTF bull + 1d HMA bull or crossover
            if htf_1w_bull:
                if hma_crossover_long and not rsi_overbought:
                    desired_signal = SIZE_STRONG
                elif hma_1d_bull and not rsi_overbought:
                    desired_signal = SIZE_BASE
            # Short: HTF bear + 1d HMA bear or crossover
            elif htf_1w_bear:
                if hma_crossover_short and not rsi_oversold:
                    desired_signal = -SIZE_STRONG
                elif hma_1d_bear and not rsi_oversold:
                    desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (45-55 CHOP): Use both strategies loosely
        else:
            # Mean reversion priority
            if crsi_oversold and not rsi_overbought:
                desired_signal = SIZE_BASE
            elif crsi_overbought and not rsi_oversold:
                desired_signal = -SIZE_BASE
            # Trend continuation backup
            elif htf_1w_bull and hma_1d_bull and not rsi_overbought:
                desired_signal = SIZE_BASE
            elif htf_1w_bear and hma_1d_bear and not rsi_oversold:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals