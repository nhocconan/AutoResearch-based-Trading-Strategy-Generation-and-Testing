#!/usr/bin/env python3
"""
Experiment #1279: 1h Primary + 4h/12h HTF — HMA Trend + cRSI Pullback

Hypothesis: Recent failures (1267-1278) show 0 trades due to overly strict entry conditions.
This strategy uses LOOSE entry thresholds to guarantee 40-80 trades/year while maintaining
3+ confluence filters. Key changes from failed attempts:

1. 4h HMA(21) for primary trend direction (smoother than 12h, more signals than 1d)
2. 12h HMA(21) for regime bias (only trade with major trend)
3. Connors RSI (CRSI) for pullback entries within trend (RSI(3) + RSI_Streak(2) + PercentRank(100))/3
4. VERY LOOSE thresholds: CRSI < 45 for long (not <10), CRSI > 55 for short (not >90)
5. ROC(5) > 0 for long confirmation (not >5), ROC(5) < 0 for short (not <-5)
6. ATR(14) 3x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30

Why this should work:
- 1h timeframe = natural 40-80 trades/year with proper filtering
- Dual HTF filter (4h+12h) = directional bias without over-filtering
- CRSI pullback = enters on dips within trend (proven 75% win rate in research)
- Loose thresholds = guarantees trades while maintaining edge
- No session filter = more trade opportunities across all symbols

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_HMA rising + 12h_HMA bullish + CRSI < 45 + ROC(5) > 0
- SHORT: 4h_HMA falling + 12h_HMA bearish + CRSI > 55 + ROC(5) < 0

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_pullback_4h12h_loose_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI Streak component of Connors RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            streak = 1
            j = i - 1
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            streak = -1
            j = i - 1
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        
        if streak >= 0:
            streak_rsi[i] = 100.0 * streak / period if streak <= period else 100.0
        else:
            streak_rsi[i] = 100.0 * streak / period if abs(streak) <= period else 0.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Percent Rank component of Connors RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pr[i]) / 3.0
    
    return crsi

def calculate_roc(close, period=5):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_5 = calculate_roc(close, period=5)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Also calculate 1h HMA for local trend
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(roc_5[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA slope + 12h HMA bias) ===
        # 4h HMA slope (compare to 2 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 2 and not np.isnan(hma_4h_aligned[i-2]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-2]
        
        # 12h HMA bias
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # 1h price vs 1h HMA for local confirmation
        price_above_1h = close[i] > hma_1h[i]
        price_below_1h = close[i] < hma_1h[i]
        
        # === MOMENTUM & MEAN REVERSION (CRSI + ROC) ===
        crsi_val = crsi[i]
        roc = roc_5[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA rising + 12h bullish + CRSI oversold + ROC positive
        if hma_4h_slope > 0 and price_above_12h and price_above_1h:
            if crsi_val < 45 and roc > 0:
                if crsi_val < 25:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h HMA falling + 12h bearish + CRSI overbought + ROC negative
        elif hma_4h_slope < 0 and price_below_12h and price_below_1h:
            if crsi_val > 55 and roc < 0:
                if crsi_val > 75:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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