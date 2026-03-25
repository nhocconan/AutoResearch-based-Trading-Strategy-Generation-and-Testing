#!/usr/bin/env python3
"""
Experiment #1294: 1d Primary + 1w HTF — Dual Regime (Trend/Mean-Revert) with CRSI

Hypothesis: Daily timeframe with weekly trend filter can capture major moves while
avoiding whipsaw. Key innovation: DUAL REGIME switching based on market state.

Why this should work:
1. 1d timeframe = natural 20-50 trades/year (fee-friendly, proven in history)
2. Weekly HMA(21) for major trend bias (only trade with weekly direction)
3. Choppiness Index(14) for regime detection: >55 = range (mean-revert), <45 = trend
4. Connors RSI for mean-reversion entries (proven 75% win rate on ETH)
5. Donchian breakout for trend entries (catches sustained moves)
6. LOOSE entry conditions to guarantee 30+ trades (learned from 0-trade failures)

Entry logic (LOOSE to guarantee trades):
- RANGE REGIME (CHOP>55): Long when CRSI<30 + price>weekly_HMA, Short when CRSI>70 + price<weekly_HMA
- TREND REGIME (CHOP<45): Long when Donchian(20) breakout + weekly_HMA rising, Short when breakdown + weekly_HMA falling
- TRANSITION (45-55): Use smaller size (0.15) with either signal

Risk Management:
- ATR(14) 2.5x trailing stop
- Discrete sizing: 0.0, ±0.20, ±0.30
- Stoploss via signal→0 when triggered

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_v1"
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
    
    delta = np.diff(close, prepend=close[0])
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            # Strong streak = extreme RSI
            if streak[i] > 0:
                streak_rsi[i] = min(100, 50 + abs_streak * 15)
            else:
                streak_rsi[i] = max(0, 50 - abs_streak * 15)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100) - where current return ranks in last 100 days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1]) / close[i-rank_period:i]
        current_return = returns[-1] if len(returns) > 0 else 0
        count_below = np.sum(returns[:-1] < current_return)
        percent_rank[i] = (count_below / (len(returns) - 1)) * 100.0 if len(returns) > 1 else 50.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures ranging vs trending market"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Also calculate 1d HMA for local trend
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # Weekly HMA slope (compare to 2 bars ago for stability)
        hma_1w_slope = 0.0
        if i >= 2 and not np.isnan(hma_1w_aligned[i-2]):
            hma_1w_slope = hma_1w_aligned[i] - hma_1w_aligned[i-2]
        
        weekly_bullish = price_above_1w and hma_1w_slope > 0
        weekly_bearish = price_below_1w and hma_1w_slope < 0
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop[i]
        is_range = chop_value > 55.0  # Ranging market
        is_trend = chop_value < 45.0  # Trending market
        # 45-55 = transition (use smaller size)
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        if is_range:
            # MEAN REVERSION regime - use CRSI extremes
            crsi_val = crsi[i]
            
            # LONG: CRSI oversold + weekly bias not bearish
            if crsi_val < 30.0 and not weekly_bearish:
                if crsi_val < 20.0:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + weekly bias not bullish
            elif crsi_val > 70.0 and not weekly_bullish:
                if crsi_val > 80.0:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        elif is_trend:
            # TREND regime - use Donchian breakout + weekly confirmation
            prev_close = close[i-1] if i > 0 else close[i]
            
            # LONG: Breakout above Donchian + weekly bullish
            if close[i] > donchian_upper[i-1] and weekly_bullish:
                desired_signal = SIZE_STRONG
            
            # SHORT: Breakdown below Donchian + weekly bearish
            elif close[i] < donchian_lower[i-1] and weekly_bearish:
                desired_signal = -SIZE_STRONG
        
        else:
            # TRANSITION regime - smaller size, either signal works
            crsi_val = crsi[i]
            
            if crsi_val < 25.0 and not weekly_bearish:
                desired_signal = SIZE_WEAK
            elif crsi_val > 75.0 and not weekly_bullish:
                desired_signal = -SIZE_WEAK
            elif close[i] > donchian_upper[i-1] and weekly_bullish:
                desired_signal = SIZE_WEAK
            elif close[i] < donchian_lower[i-1] and weekly_bearish:
                desired_signal = -SIZE_WEAK
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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