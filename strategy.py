#!/usr/bin/env python3
"""
Experiment #1394: 1d Primary + 1w HTF — Dual Regime (Trend + Mean Reversion)

Hypothesis: Daily timeframe with weekly trend filter + Connors RSI for entry timing
+ Choppiness Index regime detection will capture both trending and ranging markets.

Why this should work where others failed:
1. 1d timeframe = natural 20-50 trades/year (fee-friendly, not overtraded)
2. 1w HMA(21) for major trend bias (avoid counter-trend trades in bear markets)
3. Connors RSI (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for precise entries
4. Choppiness Index(14) regime switch: trend-follow when CHOP<50, mean-revert when CHOP>61
5. ATR(14) trailing stop for risk management

Entry Logic:
- TREND MODE (CHOP < 50): Long when price > 1w_HMA + Connors RSI < 35 (pullback)
                         Short when price < 1w_HMA + Connors RSI > 65 (rally)
- MEAN REVERSION MODE (CHOP > 61): Long when Connors RSI < 20 + price < BB_lower
                                   Short when Connors RSI > 80 + price > BB_upper

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_chop_1w_v1"
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

def calculate_rsi_streak(close, period=2):
    """Connors RSI Streak Component: RSI of up/down streak length"""
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to absolute streak for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, period)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Connors RSI Percent Rank Component: percentile of today's return vs past N days"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        window = returns[i - period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= returns[i])
            percent_rank[i] = rank / len(valid) * 100
    
    return percent_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index: measures market choppy vs trending (100*LOG10(SUM(ATR,n))/LOG10(Highest High - Lowest Low))"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.nansum(tr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum) / np.log10(price_range)
    
    return choppiness

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

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
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
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
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
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
        # CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        is_trending = choppiness[i] < 50
        is_ranging = choppiness[i] > 61
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_value = crsi[i]
        crsi_oversold = crsi_value < 35  # Pullback in uptrend
        crsi_overbought = crsi_value > 65  # Rally in downtrend
        crsi_extreme_low = crsi_value < 20  # Mean reversion long
        crsi_extreme_high = crsi_value > 80  # Mean reversion short
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # TREND MODE: Follow 1w trend, enter on Connors RSI pullback
        if is_trending:
            # LONG: 1w bullish + Connors RSI oversold (pullback entry)
            if price_above_1w and crsi_oversold:
                desired_signal = SIZE_BASE
            
            # SHORT: 1w bearish + Connors RSI overbought (rally entry)
            elif price_below_1w and crsi_overbought:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION MODE: Fade extremes at Bollinger Bands
        elif is_ranging:
            # LONG: Connors RSI extreme low + price near BB lower
            if crsi_extreme_low and close[i] <= bb_lower[i]:
                desired_signal = SIZE_BASE
            
            # SHORT: Connors RSI extreme high + price near BB upper
            elif crsi_extreme_high and close[i] >= bb_upper[i]:
                desired_signal = -SIZE_BASE
        
        # HYBRID MODE (50 <= CHOP <= 61): Use simpler logic
        else:
            # Just use 1w trend + Connors RSI extremes
            if price_above_1w and crsi_value < 40:
                desired_signal = SIZE_BASE * 0.8
            elif price_below_1w and crsi_value > 60:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if abs(desired_signal) >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG if desired_signal > 0 else -SIZE_STRONG
        elif abs(desired_signal) >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE if desired_signal > 0 else -SIZE_BASE
        elif abs(desired_signal) >= 0.15:
            final_signal = 0.15 if desired_signal > 0 else -0.15
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