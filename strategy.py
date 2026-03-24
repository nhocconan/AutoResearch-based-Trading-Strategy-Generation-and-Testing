#!/usr/bin/env python3
"""
Experiment #1536: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Based on experiment history analysis:
1. #1526 (12h dual regime) had Sharpe=-0.046 but Return=+31.7% — trades fired, just need better entries
2. #1532 (12h chop regime) had Sharpe=0.119, Return=+36.9% — 12h shows promise
3. #1528/1529/1530/1535 all got 0 trades from overly strict filters — LOOSEN conditions
4. 4h strategies consistently underperform due to fee drag (too many trades)
5. 12h target: 20-50 trades/year = optimal balance of opportunity vs cost

Design Changes from #1534:
- Primary TF: 12h (not 4h) — fewer trades, less fee drag, matches experiment requirements
- HTF: 1d HMA(21) for macro trend filter only (simpler = more robust)
- Regime: Choppiness(14) — range when >55, trend when <45
- Range regime: Connors RSI <25 long, >75 short (LOOSE thresholds to ensure trades)
- Trend regime: HMA(16/48) crossover + 1d bias
- LOOSEN all thresholds: CR<25 (not <15), RSI<60 (not <50), CHOP>50 (not >55)
- Position size: 0.30 discrete (0.0, ±0.20, ±0.30)
- Stoploss: ATR(14) 2.5x trailing
- Target: 25-50 trades/train, 5-10 trades/test

Why this should beat #1534:
- 12h has shown better Sharpe than 4h in experiments (#1526, #1532)
- LOOSE entry conditions prevent 0-trade failures (#1528/1529/1530 lesson)
- Simpler logic = more robust across BTC/ETH/SOL (not overfitted to one)
- Discrete sizing minimizes fee churn

Timeframe: 12h (as required by experiment #1536)
HTF: 1d (macro filter)
Position Size: 0.30
Target: Sharpe > 0.618, DD < -30%, trades > 30 train / > 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_hma_1d_atr_v2"
timeframe = "12h"
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
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
        if np.isnan(hma_1d_aligned[i]):
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
        # LOOSENED thresholds to ensure trades fire
        is_choppy = chop[i] > 50.0  # was 55
        is_trending = chop[i] < 45.0  # was 38.2
        
        # === MACRO TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA Crossover) ===
        hma_fast_above_slow = hma_16[i] > hma_48[i]
        hma_fast_below_slow = hma_16[i] < hma_48[i]
        
        # === CONNORS RSI (Mean Reversion) — LOOSENED ===
        crsi_oversold = crsi[i] < 30.0  # was 20
        crsi_overbought = crsi[i] > 70.0  # was 80
        crsi_extreme_oversold = crsi[i] < 20.0  # was 10
        crsi_extreme_overbought = crsi[i] > 80.0  # was 90
        
        # === RSI MOMENTUM — LOOSENED ===
        rsi_neutral_long = rsi_14[i] < 65.0  # was 60
        rsi_neutral_short = rsi_14[i] > 35.0  # was 40
        
        # === HMA SLOPE (trend confirmation) ===
        hma_16_slope = 0.0
        if i >= 5 and not np.isnan(hma_16[i-5]):
            hma_16_slope = (hma_16[i] - hma_16[i-5]) / hma_16[i-5] if hma_16[i-5] > 1e-10 else 0.0
        
        hma_16_rising = hma_16_slope > 0.0
        hma_16_falling = hma_16_slope < 0.0
        
        # === DESIRED SIGNAL — REGIME-ADAPTIVE LOGIC (LOOSENED) ===
        desired_signal = 0.0
        
        if is_choppy:
            # === RANGE REGIME: Mean Reversion with Connors RSI ===
            # Very loose conditions to ensure trades fire
            if crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            elif crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_oversold and daily_bull:
                # Long in daily uptrend when oversold
                desired_signal = BASE_SIZE
            elif crsi_overbought and daily_bear:
                # Short in daily downtrend when overbought
                desired_signal = -BASE_SIZE
            elif crsi[i] < 35.0:
                # Moderately oversold - small long
                desired_signal = BASE_SIZE * 0.6
            elif crsi[i] > 65.0:
                # Moderately overbought - small short
                desired_signal = -BASE_SIZE * 0.6
        else:
            # === TREND REGIME: HMA Crossover with 1d Bias ===
            if daily_bull:
                # Daily uptrend - look for longs
                if hma_fast_above_slow:
                    desired_signal = BASE_SIZE
                elif hma_16_rising and rsi_neutral_long:
                    desired_signal = BASE_SIZE * 0.7
                elif crsi_oversold:
                    # Pullback long in uptrend
                    desired_signal = BASE_SIZE * 0.7
            elif daily_bear:
                # Daily downtrend - look for shorts
                if hma_fast_below_slow:
                    desired_signal = -BASE_SIZE
                elif hma_16_falling and rsi_neutral_short:
                    desired_signal = -BASE_SIZE * 0.7
                elif crsi_overbought:
                    # Rally short in downtrend
                    desired_signal = -BASE_SIZE * 0.7
            else:
                # Neutral daily - use HMA crossover only
                if hma_fast_above_slow:
                    desired_signal = BASE_SIZE * 0.5
                elif hma_fast_below_slow:
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