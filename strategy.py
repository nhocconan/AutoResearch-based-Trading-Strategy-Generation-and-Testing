#!/usr/bin/env python3
"""
Experiment #490: 1h Primary + 4h/12h HTF — Choppiness Regime + CRSI + HMA Trend + ATR Stop

Hypothesis: After 400+ failed experiments, the key insight is:
1. 0-trade failures (#478-488) come from OVERLY STRICT entry conditions
2. #486 (12h HMA+ADX+RSI) works because it's SIMPLE and generates trades
3. For 1h timeframe, we MUST use HTF (4h/12h) for direction, 1h only for entry timing
4. Choppiness Index regime filter prevents wrong strategy in wrong market
5. Connors RSI (CRSI) has 75% win rate for mean reversion in range markets
6. Relaxed thresholds (CRSI < 25 / > 75, not < 10 / > 90) ensure trade generation

Strategy Logic:
- CHOP(14) > 55 = RANGE regime → use CRSI mean reversion
- CHOP(14) < 45 = TREND regime → use HMA trend following
- 4h/12h HMA for major trend bias (only take longs if 4h HMA bullish)
- 1h CRSI for entry timing in range regime
- 1h HMA pullback for entry timing in trend regime
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (smaller for 1h to reduce fee churn)
- HOLD logic to maintain positions while conditions intact

Target: Sharpe > 0.612, DD < -40%, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    if half < 1 or sqrt_period < 1:
        return hma
    
    def wma(series, w_period):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        for i in range(w_period - 1, len(series)):
            if np.any(np.isnan(series[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(series[i - w_period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # Calculate ATR for each bar
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    for i in range(period - 1, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(returns, 100)) / 3
    
    Streak: consecutive up/down days (positive for up streak, negative for down)
    PercentRank: percentile of today's return vs last 100 returns
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute values for RSI calculation)
    streak_abs = np.abs(streak)
    # For RSI of streak: gain when streak increases, loss when decreases
    streak_delta = np.diff(streak_abs)
    streak_gain = np.zeros(n)
    streak_loss = np.zeros(n)
    streak_gain[1:] = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss[1:] = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_s = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_s = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_gain_s / (streak_loss_s + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank of returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / (close[:-1] + 1e-10) * 100.0
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i - rank_period:i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < returns[i]) / len(window) * 100.0
    
    # Combine CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    hma_1h = calculate_hma(close, period=21)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Smaller size for 1h to reduce fee churn
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + buffers
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(crsi_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = RANGE (mean reversion)
        # CHOP < 45 = TREND (trend following)
        # 45-55 = transition (no new entries)
        is_range_regime = chop_1h[i] > 55.0
        is_trend_regime = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS (4h + 12h HMA) ===
        # Only take longs if HTF is bullish, shorts if HTF is bearish
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong HTF bias: both 4h and 12h agree
        htf_strong_bull = htf_4h_bullish and htf_12h_bullish
        htf_strong_bear = htf_4h_bearish and htf_12h_bearish
        
        # === 1h LOCAL TREND ===
        price_above_hma = close[i] > hma_1h[i]
        price_below_hma = close[i] < hma_1h[i]
        hma_slope_up = hma_1h[i] > hma_1h[i - 5] if i >= 5 else False
        hma_slope_down = hma_1h[i] < hma_1h[i - 5] if i >= 5 else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === RANGE REGIME: CRSI Mean Reversion ===
        if is_range_regime:
            # Long: CRSI oversold + HTF not strongly bearish
            if crsi_1h[i] < 25.0 and not htf_strong_bear:
                desired_signal = SIZE
            
            # Short: CRSI overbought + HTF not strongly bullish
            elif crsi_1h[i] > 75.0 and not htf_strong_bull:
                desired_signal = -SIZE
        
        # === TREND REGIME: HMA Trend Following ===
        elif is_trend_regime:
            # Long: HTF bullish + price above 1h HMA + HMA sloping up
            if htf_strong_bull and price_above_hma and hma_slope_up:
                desired_signal = SIZE
            
            # Short: HTF bearish + price below 1h HMA + HMA sloping down
            elif htf_strong_bear and price_below_hma and hma_slope_down:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if regime/trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if range regime with CRSI not overbought OR trend regime with HTF bullish
                if (is_range_regime and crsi_1h[i] < 70.0) or (is_trend_regime and htf_4h_bullish):
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if range regime with CRSI not oversold OR trend regime with HTF bearish
                if (is_range_regime and crsi_1h[i] > 30.0) or (is_trend_regime and htf_4h_bearish):
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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