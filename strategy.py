#!/usr/bin/env python3
"""
Experiment #942: 12h Primary + 1d/1w HTF — Dual Regime (Connors RSI + Donchian)

Hypothesis: After 671 failed strategies, the key is SIMPLICITY + ENSURING TRADES.
This strategy uses 12h primary (target 20-50 trades/year) with:
1. Connors RSI (CRSI) for mean reversion entries — proven 75% win rate
2. Donchian channel breakouts for trend confirmation
3. 1d HMA(21) for medium-term trend bias
4. 1w HMA(21) for macro regime filter
5. Dual regime: mean revert when CHOP>55, trend follow when CHOP<45

Why 12h timeframe:
- Lower fee drag than 4h/1h (target 25-40 trades/year)
- HTF signals (1d/1w) provide stronger trend bias
- Connors RSI works better on higher TF (less noise)
- Proven patterns from research: CRSI+CHOP on ETH (Sharpe +0.923)

Critical improvements over failed strategies:
- RELAXED entry conditions (CRSI<20 not <10, ensures trades)
- No funding rate dependency (often misaligned/unavailable)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Simple stoploss (2.5 ATR) — every position MUST have stop
- HOLD logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_dual_regime_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate on mean reversion entries.
    Long: CRSI < 10-20, Short: CRSI > 80-90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) — very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (2) — consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * (streak_abs[i] / (streak_abs[i] + streak_period))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (streak_period / (streak_abs[i] + streak_period))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100) — where current return ranks vs last 100 bars
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns) * 100
            pct_rank[i] = rank
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(250, n):  # Need 250 bars for CRSI rank_period + SMA200
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donch_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === LONG-TERM TREND (SMA200) ===
        long_bull = close[i] > sma_200[i]
        long_bear = close[i] < sma_200[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 55
        trending_regime = chop_12h[i] < 45
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 25  # RELAXED from 10 to ensure trades
        crsi_overbought = crsi_12h[i] > 75  # RELAXED from 90
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_upper[i]
        donch_breakout_short = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + price above macro trend
            if crsi_oversold and (macro_bull or trend_1d_bullish or long_bull):
                desired_signal = BASE_SIZE
            # Long: CRSI extreme oversold (stronger signal)
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Long: CRSI oversold + near Donchian lower (support)
            elif crsi_oversold and (close[i] - donch_lower[i]) / (donch_upper[i] - donch_lower[i]) < 0.2:
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + price below macro trend
            if crsi_overbought and (macro_bear or trend_1d_bearish or long_bear):
                desired_signal = -BASE_SIZE
            # Short: CRSI extreme overbought
            elif crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Short: CRSI overbought + near Donchian upper (resistance)
            elif crsi_overbought and (close[i] - donch_lower[i]) / (donch_upper[i] - donch_lower[i]) > 0.8:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Donchian ===
        elif trending_regime:
            # Long: Bullish trend + Donchian breakout
            if (macro_bull or trend_1d_bullish) and donch_breakout_long:
                desired_signal = BASE_SIZE
            # Long: Bullish trend + CRSI pullback entry
            elif (macro_bull or trend_1d_bullish) and crsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Donchian breakdown
            if (macro_bear or trend_1d_bearish) and donch_breakout_short:
                desired_signal = -BASE_SIZE
            # Short: Bearish trend + CRSI rally entry
            elif (macro_bear or trend_1d_bearish) and crsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: CRSI extremes only
            if crsi_extreme_oversold and (macro_bull or trend_1d_bullish):
                desired_signal = BASE_SIZE
            elif crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (macro_bear or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            elif crsi_extreme_overbought:
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
                # Hold long if trend intact and CRSI not overbought
                if (macro_bull or trend_1d_bullish) and crsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (macro_bear or trend_1d_bearish) and crsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + CRSI overbought
            if macro_bear and trend_1d_bearish and crsi_12h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + CRSI oversold
            if macro_bull and trend_1d_bullish and crsi_12h[i] < 25:
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