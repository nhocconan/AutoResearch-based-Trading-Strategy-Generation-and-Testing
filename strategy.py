#!/usr/bin/env python3
"""
Experiment #1634: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend bias provides optimal trade frequency 
(20-50/year) while avoiding noise. Connors RSI (CRSI) proven 75% win rate for mean 
reversion. Choppiness Index regime filter adapts logic to market state.

Key design choices based on failure analysis (#1626 CRSI failed):
1. LOOSE CRSI thresholds: 15/85 (not 10/90) to guarantee trades
2. RSI(3) component of CRSI is key - very responsive for daily entries
3. 1w HMA(21) for trend bias - simpler than dual 1w+1d which reduced trades
4. Choppiness regime: mean revert when CHOP>61, trend follow when CHOP<38
5. Asymmetric sizing: 0.25 base, 0.30 with trend confirmation
6. 2.5x ATR trailing stoploss via signal→0

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3 test):
- RANGE (CHOP>61): CRSI<15 long, CRSI>85 short (mean reversion)
- TREND (CHOP<38): CRSI<25 + 1w HMA bias long, CRSI>75 + 1w HMA bias short
- NEUTRAL: CRSI extremes only (15/85) with 1w HMA filter

Why this beats #1626 (Sharpe=-1.351):
- Looser CRSI thresholds (15/85 vs 10/90) = more trades
- Simpler 1w HMA bias (not complex multi-HTF)
- Better regime thresholds (61/38 not 61.8/38.2)
- Discrete signal sizes reduce fee churn

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_loose_v2"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Current close vs past 100 days
    
    CRSI < 10 = extreme oversold (long), CRSI > 90 = extreme overbought (short)
    Proven 75% win rate for mean reversion
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) - very responsive
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # RSI(Streak) - consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_gain[i-streak_period+1:i+1])
        avg_loss = np.mean(streak_loss[i-streak_period+1:i+1])
        if avg_loss > 0:
            streak_rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            streak_rsi[i] = 100
    
    # PercentRank - current close vs past 100 closes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100 * count_below / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    
    # Warmup period
    min_bars = 120
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
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
        chop = chop_14[i]
        is_trend_regime = chop < 38.0
        is_range_regime = chop > 61.0
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trades) ===
        crsi_val = crsi[i]
        
        # CRSI extremes - LOOSE for more trades
        crsi_oversold = crsi_val < 25
        crsi_overbought = crsi_val > 75
        crsi_extreme_low = crsi_val < 15
        crsi_extreme_high = crsi_val > 85
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        if is_range_regime:
            # LONG: CRSI extreme oversold
            if crsi_extreme_low:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI extreme overbought
            elif crsi_extreme_high:
                desired_signal = -SIZE_BASE
            
            # Moderate extremes with trend bias
            elif crsi_oversold and price_above_1w:
                desired_signal = SIZE_BASE
            elif crsi_overbought and price_below_1w:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Follow trend with CRSI pullback entries
        elif is_trend_regime:
            # LONG: 1w bullish + CRSI pullback (not extreme)
            if price_above_1w and crsi_oversold and crsi_val > 10:
                desired_signal = SIZE_STRONG
            elif price_above_1w and crsi_val < 40:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + CRSI pullback (not extreme)
            elif price_below_1w and crsi_overbought and crsi_val < 90:
                desired_signal = -SIZE_STRONG
            elif price_below_1w and crsi_val > 60:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: CRSI extremes only with 1w filter
        else:
            # LONG: CRSI oversold + 1w not bearish
            if crsi_oversold and not price_below_1w:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + 1w not bullish
            elif crsi_overbought and not price_above_1w:
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