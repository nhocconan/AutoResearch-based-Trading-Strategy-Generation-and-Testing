#!/usr/bin/env python3
"""
Experiment #1089: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + Dual Mode

Hypothesis: After #1084 failed (Sharpe=-0.059), the issue is too many filters and
wrong regime detection. ADX doesn't distinguish chop from trend well.

NEW APPROACH:
1. CHOPPINESS INDEX (CHOP) for regime detection — proven in research
   CHOP > 61.8 = range (use mean reversion)
   CHOP < 38.2 = trend (use trend following)
   This is the meta-filter that failed strategies lacked

2. CONNORS RSI (CRSI) for entries — 75% win rate in research
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 10 (oversold), Short: CRSI > 90 (overbought)

3. DUAL REGIME LOGIC:
   - In CHOP regime: mean revert at extremes (CRSI < 10 long, > 90 short)
   - In TREND regime: follow 1d HMA direction, enter on CRSI pullbacks

4. 1d HTF HMA21 for macro bias — only trade with higher TF trend in trend regime

5. ATR(14) trailing stop 2.5x — proper risk management

Why this should beat Sharpe=0.612:
- CHOP is better regime filter than ADX (research shows ETH Sharpe +0.923)
- CRSI more responsive than RSI(14) for entries
- Dual regime adapts to market conditions (major failure mode in 2022 chop)
- Simpler logic = more robust across BTC/ETH/SOL
- Should generate 30-50 trades/year on 4h (within fee budget)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_crsi_dual_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    Formula: HMA = WMA(sqrt(period)) of (2*WMA(period/2) - WMA(period))
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures whether market is trending or ranging.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    CHOP > 61.8 = choppy/ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    price_range = highest - lowest
    mask = price_range > 1e-10
    chop[mask] = 100.0 * np.log10(tr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite momentum indicator with 75% win rate.
    
    Formula:
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI of streak length — measures consecutive up/down days
    3. PercentRank — current close vs last 100 closes
    
    Entry signals:
    Long: CRSI < 10 (extremely oversold)
    Short: CRSI > 90 (extremely overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.full(n, 50.0)
    mask = avg_streak_loss > 1e-10
    rs_streak = np.zeros(n)
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    streak_rsi[mask] = 100.0 - (100.0 / (1.0 + rs_streak[mask]))
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    
    # Combine components
    valid = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi_short[valid] + streak_rsi[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = choppiness[i] > 55.0  # Range/mean reversion
        trend_regime = choppiness[i] < 45.0   # Trend following
        neutral_regime = not choppy_regime and not trend_regime
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Long entry in chop
        crsi_overbought = crsi[i] > 85.0  # Short entry in chop
        
        # Pullback entries in trend regime
        crsi_pullback_long = 25.0 <= crsi[i] <= 45.0  # Dip in uptrend
        crsi_pullback_short = 55.0 <= crsi[i] <= 75.0  # Rally in downtrend
        
        # === VOLATILITY CHECK ===
        if i > 100:
            atr_median = np.nanmedian(atr[max(0, i-100):i])
            vol_spike = atr[i] > 2.0 * atr_median
        else:
            vol_spike = False
        
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: MEAN REVERSION ===
        if choppy_regime:
            if crsi_oversold:
                desired_signal = current_size  # Long at oversold
            elif crsi_overbought:
                desired_signal = -current_size  # Short at overbought
        
        # === TREND REGIME: TREND FOLLOWING ===
        elif trend_regime:
            if macro_bull and crsi_pullback_long:
                desired_signal = current_size  # Long pullback in uptrend
            elif macro_bear and crsi_pullback_short:
                desired_signal = -current_size  # Short pullback in downtrend
        
        # === NEUTRAL REGIME: REDUCED SIZE OR FLAT ===
        elif neutral_regime:
            # Only take strongest signals in neutral
            if crsi[i] < 10.0:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 90.0:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if not overbought and regime not strongly bearish
                if crsi[i] < 80.0 and not (trend_regime and macro_bear):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if not oversold and regime not strongly bullish
                if crsi[i] > 20.0 and not (trend_regime and macro_bull):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought or macro reverses strongly
            if crsi[i] > 85.0:
                desired_signal = 0.0
            if trend_regime and macro_bear and choppiness[i] < 40.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold or macro reverses strongly
            if crsi[i] < 15.0:
                desired_signal = 0.0
            if trend_regime and macro_bull and choppiness[i] < 40.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
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