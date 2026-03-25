#!/usr/bin/env python3
"""
Experiment #1292: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI

Hypothesis: Based on experiment #1284 (12h dual regime chop crsi) which achieved 
Sharpe=0.054 and #1286 (1d version) with Sharpe=0.148, combining Choppiness Index
regime detection with Connors RSI entries shows promise for BTC/ETH which struggle
with pure trend strategies.

Key innovations vs failed strategies:
1. CHOPPINESS INDEX (14) regime switch: CHOP>61.8=range (mean revert), CHOP<38.2=trend
2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - more sensitive
3. 1d HMA(21) for major trend bias (only trade with daily direction)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE CRSI thresholds to guarantee 30-60 trades/year on 12h

Why this should beat KAMA+ROC (Sharpe=0.447):
- Dual regime adapts to BTC/ETH behavior (trending 2021, ranging 2022-2024)
- Connors RSI catches reversals better than standard RSI in bear markets
- 12h timeframe = natural 20-50 trades/year (fee-friendly)
- Works in both bull and bear regimes (unlike pure trend following)

Entry logic (LOOSE to guarantee trades):
- RANGE (CHOP>61.8): LONG if CRSI<20 + price>1d_HMA, SHORT if CRSI>80 + price<1d_HMA
- TREND (CHOP<38.2): LONG if CRSI<35 + 1d_HMA rising, SHORT if CRSI>65 + 1d_HMA falling

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_hma_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) * 100 / log10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * (atr_sum / price_range) / np.log10(period)
    
    return chop

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
        elif avg_gain[i] > 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 50.0
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Counts consecutive up/down days, converts to RSI-like score
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like 0-100 scale
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        up_streaks = 0
        down_streaks = 0
        for j in range(i-period+1, i+1):
            if streak[j] > 0:
                up_streaks += streak[j]
            elif streak[j] < 0:
                down_streaks += abs(streak[j])
        
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Percentage of closes in lookback period that are lower than current close
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])  # exclude current
        pr[i] = 100.0 * count_lower / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) indicate reversal opportunities
    """
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(len(close), np.nan, dtype=np.float64)
    for i in range(len(close)):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    
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
    
    # Warmup period (need 100 bars for CRSI percent rank)
    min_bars = 100
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_range = chop > 61.8  # Choppy/ranging market
        is_trend = chop < 38.2  # Trending market
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1d HMA slope (compare to 5 bars ago for stability on daily)
        hma_1d_slope = 0.0
        if i >= 5 and not np.isnan(hma_1d_aligned[i-5]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME (mean reversion at extremes)
        if is_range:
            # LONG: CRSI oversold + price above daily HMA (bullish bias)
            if crsi[i] < 25 and price_above_1d:
                if crsi[i] < 15:
                    desired_signal = SIZE_STRONG  # Very oversold
                else:
                    desired_signal = SIZE_BASE  # Moderately oversold
            
            # SHORT: CRSI overbought + price below daily HMA (bearish bias)
            elif crsi[i] > 75 and price_below_1d:
                if crsi[i] > 85:
                    desired_signal = -SIZE_STRONG  # Very overbought
                else:
                    desired_signal = -SIZE_BASE  # Moderately overbought
        
        # TREND REGIME (follow the trend on pullbacks)
        elif is_trend:
            # LONG: Trending up + CRSI pullback
            if hma_1d_slope > 0 and crsi[i] < 40:
                if crsi[i] < 25:
                    desired_signal = SIZE_STRONG  # Deep pullback
                else:
                    desired_signal = SIZE_BASE  # Shallow pullback
            
            # SHORT: Trending down + CRSI bounce
            elif hma_1d_slope < 0 and crsi[i] > 60:
                if crsi[i] > 75:
                    desired_signal = -SIZE_STRONG  # Strong bounce
                else:
                    desired_signal = -SIZE_BASE  # Weak bounce
        
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