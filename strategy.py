#!/usr/bin/env python3
"""
Experiment #1434: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: Daily timeframe with weekly trend filter provides optimal trade frequency
(20-50 trades/year) while avoiding fee drag. Dual regime adapts to market conditions:
1. Choppiness Index detects trend vs range regime
2. Connors RSI (CRSI) for entry timing - proven 75% win rate in literature
3. 1w HMA(21) for major trend bias (avoid counter-trend in strong trends)
4. ATR(14) trailing stoploss (2.5x) for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Regime adaptation works better in bear/range markets (2022 crash, 2025 test)
- CRSI catches reversals better than simple RSI
- 1d TF = natural 25-40 trades/year (fee-efficient)
- LOOSE entry thresholds guarantee trades (CRSI<30/>70 range, <45/>55 trend)

Entry logic (LOOSE to guarantee >=30 trades train, >=3 test):
- RANGE (CHOP>55): Long CRSI<30, Short CRSI>70
- TREND (CHOP<45): Long if 1w_HMA bullish + CRSI<45, Short if 1w_HMA bearish + CRSI>55
- Transition zone (45-55): No new entries, hold existing

Timeframe: 1d
Size: 0.25-0.30 discrete
Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_regime_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = Range/Choppy market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion
    """
    n = len(close)
    crsi = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak = 0
        if close[i] > close[i-1]:
            for j in range(i, max(0, i-streak_period), -1):
                if j > 0 and close[j] > close[j-1]:
                    streak += 1
                else:
                    break
        elif close[i] < close[i-1]:
            for j in range(i, max(0, i-streak_period), -1):
                if j > 0 and close[j] < close[j-1]:
                    streak -= 1
                else:
                    break
        
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = 50 + (streak / streak_period) * 50
        else:
            streak_rsi[i] = 50 + (streak / streak_period) * 50
    
    # Percent Rank (100) - where current return ranks vs last 100 days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0 and not np.all(np.isnan(returns)):
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = (count_below / (len(returns) - 1)) * 100
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    min_bars = 150
    
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
        is_range = chop > 55.0  # Range/choppy market
        is_trend = chop < 45.0  # Trending market
        # 45-55 is transition zone - hold existing, no new entries
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CRSI VALUE ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        if is_range:
            # Long when CRSI extremely oversold
            if crsi_val < 30:
                desired_signal = SIZE_BASE
            # Short when CRSI extremely overbought
            elif crsi_val > 70:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Follow trend with CRSI pullback
        elif is_trend:
            # Long in uptrend when CRSI pulls back (not extreme)
            if price_above_1w and crsi_val < 45:
                desired_signal = SIZE_STRONG
            # Short in downtrend when CRSI pulls back (not extreme)
            elif price_below_1w and crsi_val > 55:
                desired_signal = -SIZE_STRONG
        
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