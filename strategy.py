#!/usr/bin/env python3
"""
Experiment #1306: 1d Primary + 1w HTF — Dual Regime Breakout + Mean Reversion

Hypothesis: Daily timeframe with weekly trend filter provides optimal trade frequency
(20-50/year) while maintaining signal quality. This strategy combines:

1. 1w HMA(21) for major trend bias (only trade with weekly direction)
2. 1d Regime detection via Choppiness Index (CHOP > 61.8 = range, < 38.2 = trend)
3. TREND regime: Donchian(20) breakout + RSI(14) confirmation
4. RANGE regime: Connors RSI mean reversion (CRSI < 15 long, > 85 short)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- 1d timeframe = natural 20-50 trades/year (fee-friendly)
- Weekly filter = strong directional bias without over-filtering
- Dual regime = adapts to market conditions (trend vs chop)
- Connors RSI = proven 75% win rate on mean reversion
- Loose enough entries to guarantee 30+ trades on train

Entry logic:
- TREND regime (CHOP < 38.2): Donchian breakout + RSI 40-60 + weekly alignment
- RANGE regime (CHOP > 61.8): CRSI extremes (<15 long, >85 short) + weekly bias
- TRANSITION (38.2-61.8): No trades, wait for clarity

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_donchian_crsi_1w_v1"
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
    
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0 and not np.isnan(avg_gain[i]) and not np.isnan(avg_loss[i]):
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        elif avg_loss[i] == 0 and not np.isnan(avg_gain[i]):
            rsi[i] = 100.0
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - streak of up/down days
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        else:
            streak_rsi[i] = 50.0 + (streak[i] / streak_period) * 50.0
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank (100) - where current return ranks in last 100 days
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        returns = np.zeros(rank_period, dtype=np.float64)
        for j in range(rank_period):
            if close[i - rank_period + j] != 0:
                returns[j] = (close[i - rank_period + j + 1] - close[i - rank_period + j]) / close[i - rank_period + j]
        
        current_return = (close[i] - close[i-1]) / close[i-1] if close[i-1] != 0 else 0
        count_below = np.sum(returns[:-1] < current_return)
        pct_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
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
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # 1d HMA for local trend
    hma_1d = calculate_hma(close, period=21)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND BIAS ===
        # Weekly HMA slope (compare to 2 bars ago for stability)
        hma_1w_slope = 0.0
        if i >= 2 and not np.isnan(hma_1w_aligned[i-2]):
            hma_1w_slope = hma_1w_aligned[i] - hma_1w_aligned[i-2]
        
        weekly_bullish = hma_1w_slope > 0
        weekly_bearish = hma_1w_slope < 0
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        
        # TREND regime: CHOP < 38.2
        # RANGE regime: CHOP > 61.8
        # TRANSITION: 38.2 - 61.8 (no trades)
        
        in_trend_regime = chop < 38.2
        in_range_regime = chop > 61.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if in_trend_regime:
            # TREND FOLLOWING: Donchian breakout + RSI confirmation + weekly alignment
            rsi = rsi_14[i]
            
            # LONG: Break above Donchian + RSI 40-70 + weekly bullish
            if close[i] > donchian_upper[i] and 40 <= rsi <= 70 and weekly_bullish:
                if rsi > 55:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: Break below Donchian + RSI 30-60 + weekly bearish
            elif close[i] < donchian_lower[i] and 30 <= rsi <= 60 and weekly_bearish:
                if rsi < 45:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        elif in_range_regime:
            # MEAN REVERSION: Connors RSI extremes + weekly bias filter
            if not np.isnan(crsi[i]):
                crsi_val = crsi[i]
                
                # LONG: CRSI < 15 (oversold) + weekly not strongly bearish
                if crsi_val < 15 and not weekly_bearish:
                    if crsi_val < 10:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                
                # SHORT: CRSI > 85 (overbought) + weekly not strongly bullish
                elif crsi_val > 85 and not weekly_bullish:
                    if crsi_val > 90:
                        desired_signal = -SIZE_STRONG
                    else:
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