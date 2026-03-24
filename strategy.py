#!/usr/bin/env python3
"""
Experiment #1533: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After 1140+ failed strategies, the pattern is clear:
1. 1d timeframe with 1w HTF worked in #1527 (Sharpe=0.254) — build on this
2. Current best uses Donchian+HMA+RSI (Sharpe=0.618) — need different edge
3. Research shows Connors RSI + Choppiness = ETH Sharpe +0.923
4. Dual-regime (mean-revert in chop, trend-follow otherwise) adapts to 2022 crash + 2025 bear
5. LOOSE thresholds critical — #1528/1529/1530 failed with 0 trades due to tight filters
6. 1d TF should generate 20-50 trades/year (40-100 train, 10-25 test)

Design:
- Choppiness Index(14) = regime detector (CHOP>55=range, CHOP<45=trend)
- Range regime: Connors RSI <25 long, >75 short (mean reversion, loose thresholds)
- Trend regime: HMA(16/48) crossover + 1w HMA bias (trend following)
- 1w HMA(21) = weekly macro filter (only confirms, doesn't block)
- ATR(14) 2.5x = trailing stoploss
- Position size: 0.30 (discrete: 0.0, ±0.30)
- Target: 40-100 trades/train, 10-25 trades/test

Why this should work:
- Choppiness adapts to market state (critical for 2022 crash + 2025 bear)
- Connors RSI proven mean-reversion (75% win rate in research)
- 1w HMA bias prevents major counter-trend trades
- LOOSE conditions ensure trades fire (learned from 0-trade failures)
- 1d TF = fewer trades = less fee drag than lower TFs

Timeframe: 1d (as required by experiment #1533)
HTF: 1w (weekly macro trend filter)
Position Size: 0.30 (conservative for daily volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 40
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_hma_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CR < 10 = oversold (long signal)
    CR > 90 = overbought (short signal)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + streak_abs[i] / max(streak_period, 1)))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 / (1.0 + streak_abs[i] / max(streak_period, 1))
        else:
            streak_rsi[i] = 50.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = (count_below / rank_period) * 100.0
    
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_3[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for weekly bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === MACRO TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d HMA Crossover) ===
        hma_fast_above_slow = hma_16[i] > hma_48[i]
        hma_fast_below_slow = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI MOMENTUM ===
        rsi_neutral_long = rsi_14[i] < 60.0
        rsi_neutral_short = rsi_14[i] > 40.0
        
        # === HMA SLOPE (trend confirmation) ===
        hma_16_slope = 0.0
        if i >= 5 and not np.isnan(hma_16[i-5]):
            hma_16_slope = (hma_16[i] - hma_16[i-5]) / hma_16[i-5] if hma_16[i-5] > 1e-10 else 0.0
        
        hma_16_rising = hma_16_slope > 0.0
        hma_16_falling = hma_16_slope < 0.0
        
        # === DESIRED SIGNAL — REGIME-ADAPTIVE LOGIC ===
        desired_signal = 0.0
        
        if is_choppy:
            # === RANGE REGIME: Mean Reversion with Connors RSI ===
            if weekly_bull and crsi_oversold:
                desired_signal = BASE_SIZE
            elif weekly_bear and crsi_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_extreme_oversold:
                desired_signal = BASE_SIZE * 0.7
            elif crsi_extreme_overbought:
                desired_signal = -BASE_SIZE * 0.7
        elif is_trending:
            # === TREND REGIME: HMA Crossover with 1w Bias ===
            if weekly_bull:
                if hma_fast_above_slow and rsi_neutral_long:
                    desired_signal = BASE_SIZE
                elif hma_16_rising:
                    desired_signal = BASE_SIZE * 0.7
            elif weekly_bear:
                if hma_fast_below_slow and rsi_neutral_short:
                    desired_signal = -BASE_SIZE
                elif hma_16_falling:
                    desired_signal = -BASE_SIZE * 0.7
        else:
            # === NEUTRAL REGIME: Use HMA crossover only ===
            if hma_fast_above_slow and weekly_bull:
                desired_signal = BASE_SIZE * 0.5
            elif hma_fast_below_slow and weekly_bear:
                desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.35:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals