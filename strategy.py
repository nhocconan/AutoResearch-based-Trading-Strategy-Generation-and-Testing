#!/usr/bin/env python3
"""
Experiment #1022: 12h Primary + 1d/1w HTF — Dual Regime Choppiness + Connors RSI + Donchian

Hypothesis: After analyzing 741+ failed strategies, the pattern is clear:
- 12h timeframe generates 20-50 trades/year (optimal fee/trade balance)
- Dual regime (chop vs trend) adapts to bear/range markets in 2025 test period
- Connors RSI (not standard RSI) has 75% win rate for mean reversion
- Donchian breakout confirms trend entries (reduces false signals)
- 1d HMA21 + 1w HMA21 provide macro trend bias (asymmetric long/short)

Key improvements over failed experiments:
- RELAXED entry thresholds to ensure >=30 trades on train (many failed with 0 trades)
- Connors RSI components: RSI(3) + RSI_Streak(2) + PercentRank(100)
- Choppiness Index regime: CHOP>61.8=range (mean revert), CHOP<38.2=trend (breakout)
- Donchian(20) breakout confirmation for trend entries only
- ATR(14)*2.5 trailing stop for risk management
- Discrete signal sizes: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why 12h works better than 4h:
- Less noise, fewer false signals
- Lower fee drag (20-50 trades/year vs 50-100 on 4h)
- Proven in exp#1012 (Sharpe=0.074) and exp#1013 (Sharpe=0.324)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): RSI of consecutive up/down days
    3. PercentRank(100): Percentile of today's return vs last 100 days
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    Proven 75% win rate for mean reversion
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 10:
        return crsi
    
    # Calculate returns
    returns = np.zeros(n)
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    # RSI(3)
    rsi_short = np.full(n, np.nan)
    for i in range(rsi_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i-rsi_period+1, i+1):
            if j > 0:
                change = close[j] - close[j-1]
                if change > 0:
                    gains += change
                else:
                    losses -= change
        if losses == 0:
            rsi_short[i] = 100.0
        else:
            rs = gains / losses
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak(2)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    for i in range(streak_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i-streak_period+1, i+1):
            if j > 0:
                change = streak[j] - streak[j-1]
                if change > 0:
                    gains += change
                elif change < 0:
                    losses -= change
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / losses
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        count_lower = 0
        for j in range(i-pr_period, i):
            if returns[j] < returns[i]:
                count_lower += 1
        percent_rank[i] = (count_lower / pr_period) * 100.0
    
    # Combine into CRSI
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range using EMA smoothing."""
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

def calculate_hma(series, period):
    """Hull Moving Average for trend detection."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope (positive = uptrend)."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            if hma_values[i-lookback] != 0:
                slope[i] = (hma_values[i] - hma_values[i-lookback]) / abs(hma_values[i-lookback])
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA21 for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA21 for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # HMA21 on primary for trend confirmation
    hma_12h = calculate_hma(close, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h, lookback=3)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_12h[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === MACRO TREND BIAS (HTF HMA21) ===
        # Asymmetric: long when price > 1d HMA, short when price < 1w HMA
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_12h[i] > 61.8  # Ranging → mean reversion
        regime_trend = chop_12h[i] < 38.2  # Trending → breakout
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_12h[i] < 15  # Relaxed from 10 for more trades
        crsi_overbought = crsi_12h[i] > 85  # Relaxed from 90
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above prior high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below prior low
        
        # === HMA SLOPE CONFIRMATION ===
        hma_bullish = not np.isnan(hma_12h_slope[i]) and hma_12h_slope[i] > 0.001
        hma_bearish = not np.isnan(hma_12h_slope[i]) and hma_12h_slope[i] < -0.001
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_range and macro_bull:
            # Mean reversion in ranging market with bullish bias
            if crsi_oversold:
                desired_signal = BASE_SIZE
        elif regime_trend and macro_bull:
            # Trend following in trending market
            if donchian_breakout_long and hma_bullish:
                desired_signal = BASE_SIZE
            elif crsi_oversold and hma_bullish:
                # Pullback entry in uptrend
                desired_signal = REDUCED_SIZE
        elif not regime_range and not regime_trend:
            # Neutral regime - use CRSI extremes only
            if crsi_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_range and macro_bear:
            # Mean reversion in ranging market with bearish bias
            if crsi_overbought:
                desired_signal = -BASE_SIZE
        elif regime_trend and macro_bear:
            # Trend following in trending market
            if donchian_breakout_short and hma_bearish:
                desired_signal = -BASE_SIZE
            elif crsi_overbought and hma_bearish:
                # Pullback entry in downtrend
                desired_signal = -REDUCED_SIZE
        elif not regime_range and not regime_trend:
            # Neutral regime - use CRSI extremes only
            if crsi_overbought and macro_bear:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bullish and CRSI not extreme overbought
                if macro_bull and crsi_12h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not extreme oversold
                if macro_bear and crsi_12h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses bearish
            if not macro_bull and crsi_12h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses bullish
            if not macro_bear and crsi_12h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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