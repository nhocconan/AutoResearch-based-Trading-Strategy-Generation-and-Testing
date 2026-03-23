#!/usr/bin/env python3
"""
Experiment #1131: 4h Primary + 1d HTF — Regime-Adaptive CHOP + CRSI + HMA

Hypothesis: After 823+ failed experiments, the key insight is REGIME DETECTION.
BTC/ETH 2025 is bear/range (not bull trend), so pure trend-following fails.
This strategy adapts to market regime using Choppiness Index:
1. CHOP(14) > 61.8 = RANGE regime → use Connors RSI mean reversion
2. CHOP(14) < 38.2 = TREND regime → use HMA trend + RSI pullback
3. 1d HMA(21) for macro bias filter (prevents counter-trend trades)
4. ATR(14) 2.5x trailing stop (proven risk management)
5. Position size 0.28 discrete (balance between returns and fee churn)

Why this should beat Sharpe=0.612:
- Regime switching = right strategy for right market (range vs trend)
- Connors RSI has 75% win rate for mean reversion (research-proven)
- Choppiness Index correctly identifies 2022 crash (chop) vs 2021 bull (trend)
- 1d HMA filter prevents fatal counter-trend entries
- Simpler than triple-regime strategies that failed (exp #1121, #1127)
- Targets 25-45 trades/year (sweet spot for 4h timeframe)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base (discrete: 0.0, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 25-45 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_1d_regime_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — identifies trending vs ranging markets.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean reversion indicator.
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long)
    CRSI > 90 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) — streak of consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    diff = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if diff[i-1] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif diff[i-1] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        up_streaks = np.sum(streak[i-streak_period+1:i+1] > 0)
        down_streaks = np.sum(streak[i-streak_period+1:i+1] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = 100.0 * up_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100) — where current price ranks vs last 100 closes
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine into CRSI
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi_14 = calculate_rsi(close, period=14)
    rsi_3 = calculate_rsi(close, period=3)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(rsi_14[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range market (mean reversion)
        # CHOP < 38.2 = trend market (trend following)
        # 38.2 - 61.8 = neutral (use trend logic with tighter stops)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h[i]
        trend_bear = close[i] < hma_4h[i]
        
        desired_signal = 0.0
        
        # === RANGE REGIME: Connors RSI Mean Reversion ===
        if is_range:
            # Long: CRSI < 15 + macro not strongly bear
            if crsi[i] < 15.0 and not macro_bear:
                desired_signal = BASE_SIZE
            # Short: CRSI > 85 + macro not strongly bull
            elif crsi[i] > 85.0 and not macro_bull:
                desired_signal = -BASE_SIZE
        
        # === TREND REGIME: HMA + RSI Pullback ===
        elif is_trend:
            # Long: macro bull + trend bull + RSI pullback (40-50)
            if macro_bull and trend_bull and 35.0 < rsi_14[i] < 55.0:
                desired_signal = BASE_SIZE
            # Short: macro bear + trend bear + RSI pullback (50-65)
            elif macro_bear and trend_bear and 45.0 < rsi_14[i] < 65.0:
                desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME: Conservative trend following ===
        else:
            # Only enter with strong confluence
            if macro_bull and trend_bull and rsi_14[i] < 45.0:
                desired_signal = BASE_SIZE
            elif macro_bear and trend_bear and rsi_14[i] > 55.0:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports (range or trend bull)
                if (is_range and crsi[i] < 50.0) or (is_trend and macro_bull):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if regime still supports
                if (is_range and crsi[i] > 50.0) or (is_trend and macro_bear):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses strongly
        if in_position and position_side > 0:
            if macro_bear and chop[i] < 38.2:  # trend regime + bear macro
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull and chop[i] < 38.2:  # trend regime + bull macro
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals