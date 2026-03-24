#!/usr/bin/env python3
"""
Experiment #1534: 4h Primary + 12h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: After analyzing failed experiments #1528-1533, the pattern is clear:
1. 4h timeframe has shown promise in research notes (SOL Sharpe +0.879 with HMA+RSI)
2. Connors RSI proven for ETH mean reversion (research notes: Sharpe +0.923)
3. Choppiness Index excellent regime detector (CHOP>61.8=range, CHOP<38.2=trend)
4. 12h HMA provides trend bias without being too slow (like 1d)
5. 1d HMA provides macro filter to avoid counter-trend trades
6. LOOSE entry conditions critical — #1528/1529/1530 got 0 trades from strict filters

Design:
- Primary: 4h timeframe (balances trade frequency vs fee drag)
- HTF: 12h HMA(21) for trend bias, 1d HMA(21) for macro filter
- Regime: Choppiness(14) — range when >55, trend when <45
- Range regime: Connors RSI <20 long, >80 short (mean reversion)
- Trend regime: HMA(16/48) crossover + 12h bias (trend following)
- Stoploss: ATR(14) 2.5x trailing
- Position size: 0.30 discrete (0.0, ±0.30)
- Target: 30-60 trades/train, 8-15 trades/test

Why this should work:
- 4h TF = more opportunities than 12h while keeping fee drag low
- Choppiness adapts to market state (proven in research)
- Connors RSI has 75% win rate for mean reversion
- 12h/1d HMA bias prevents counter-trend trades in strong trends
- LOOSE thresholds ensure trades fire (learned from 0-trade failures)
- Discrete sizing minimizes fee churn on signal changes

Timeframe: 4h (as required by experiment #1534)
HTF: 12h (trend bias), 1d (macro filter)
Position Size: 0.30 (conservative for 4h volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_hma_12h1d_atr_v1"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
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
        
        # === MACRO TREND BIAS (12h HMA) ===
        twelveh_bull = close[i] > hma_12h_aligned[i]
        twelveh_bear = close[i] < hma_12h_aligned[i]
        
        # === DAILY FILTER (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA Crossover) ===
        hma_fast_above_slow = hma_16[i] > hma_48[i]
        hma_fast_below_slow = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI (Mean Reversion) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI MOMENTUM ===
        rsi_neutral_long = rsi_14[i] < 65.0
        rsi_neutral_short = rsi_14[i] > 35.0
        
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
            if crsi_extreme_oversold:
                # Very oversold - long regardless of trend
                desired_signal = BASE_SIZE
            elif crsi_extreme_overbought:
                # Very overbought - short regardless of trend
                desired_signal = -BASE_SIZE
            elif twelveh_bull and crsi_oversold:
                # Long in uptrend when oversold
                desired_signal = BASE_SIZE
            elif twelveh_bear and crsi_overbought:
                # Short in downtrend when overbought
                desired_signal = -BASE_SIZE
            elif daily_bull and crsi[i] < 25.0:
                # Daily uptrend + moderately oversold
                desired_signal = BASE_SIZE * 0.7
            elif daily_bear and crsi[i] > 75.0:
                # Daily downtrend + moderately overbought
                desired_signal = -BASE_SIZE * 0.7
        else:
            # === TREND REGIME: HMA Crossover with HTF Bias ===
            if twelveh_bull and daily_bull:
                # Strong long bias (both 12h and 1d bullish)
                if hma_fast_above_slow:
                    desired_signal = BASE_SIZE
                elif hma_16_rising and rsi_neutral_long:
                    desired_signal = BASE_SIZE * 0.7
            elif twelveh_bear and daily_bear:
                # Strong short bias (both 12h and 1d bearish)
                if hma_fast_below_slow:
                    desired_signal = -BASE_SIZE
                elif hma_16_falling and rsi_neutral_short:
                    desired_signal = -BASE_SIZE * 0.7
            elif twelveh_bull:
                # 12h bullish only
                if hma_fast_above_slow:
                    desired_signal = BASE_SIZE * 0.7
            elif twelveh_bear:
                # 12h bearish only
                if hma_fast_below_slow:
                    desired_signal = -BASE_SIZE * 0.7
        
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