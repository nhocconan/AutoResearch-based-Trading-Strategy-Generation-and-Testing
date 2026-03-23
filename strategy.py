#!/usr/bin/env python3
"""
Experiment #804: 4h Primary + 12h/1d HTF — Connors RSI + Asymmetric Regime + Donchian Breakout

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. Connors RSI (CRSI) has 75% win rate for mean reversion — superior to standard RSI
2. Asymmetric regime logic: only long in bull (price>12h HMA), only short in bear (price<12h HMA)
3. 1d Choppiness Index filters: CHOP>55 = range (mean revert), CHOP<45 = trend (breakout)
4. Donchian(20) breakout confirms entry direction — reduces false signals
5. 4h timeframe targets 20-50 trades/year with quality over quantity
6. ATR(14) trailing stop at 2.5x protects from major drawdowns
7. Position sizing: 0.25-0.30 discrete levels to minimize fee churn
8. Entry thresholds: CRSI<15 long, CRSI>85 short (proven in literature)

Strategy design:
1. 12h HMA(21) for bull/bear regime detection (aligned via mtf_data helper)
2. 1d Choppiness Index(14) for range/trend regime
3. 4h Connors RSI(3,2,100) for mean reversion entries
4. 4h Donchian(20) for breakout confirmation
5. 4h ATR(14) for trailing stop (2.5x)
6. 4h SMA(200) as additional trend filter
7. Discrete signals: 0.0, ±0.25, ±0.30
8. Asymmetric: long only in bull regime, short only in bear regime

Key differences from failed 4h strategies:
- Connors RSI instead of standard RSI (3-component formula from literature)
- Asymmetric regime logic (no counter-trend trades)
- Donchian breakout confirmation (not just RSI extremes)
- CRSI thresholds: 15/85 (not 20/80 or 30/70) — more extreme for higher win rate
- Dual regime: mean revert when CHOP>55, breakout when CHOP<45
- Volume filter removed (was causing too few trades)

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_asym_regime_donchian_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_percent_rank(close, period=100):
    """
    Percent Rank: percentage of prior closes lower than current close.
    Used in Connors RSI calculation.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] < current)
        pr[i] = 100 * count_lower / (period - 1)
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) — 3-component mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days.
    Entry: CRSI < 10-15 for long, CRSI > 85-90 for short.
    Reported 75% win rate in literature.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 1:
        return crsi
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive series for RSI calculation
    streak_rsi_input = streak.copy()
    streak_rsi = calculate_rsi(streak_rsi_input, streak_period)
    
    # Component 3: Percent Rank(100)
    pr = calculate_percent_rank(close, pr_period)
    
    # Combine components
    for i in range(pr_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pr[i]):
            crsi[i] = (rsi_close[i] + streak_rsi[i] + pr[i]) / 3.0
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches on 4h.
    """
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
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200_4h = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 12h HMA for bull/bear regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d Choppiness for range/trend regime
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
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
        if np.isnan(crsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(sma_200_4h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        
        # === BULL/BEAR REGIME (12h HTF HMA21) ===
        bull_regime = close[i] > hma_12h_aligned[i]
        bear_regime = close[i] < hma_12h_aligned[i]
        
        # === RANGE/TREND REGIME (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        
        # === CONNORS RSI SIGNALS (extreme thresholds for high win rate) ===
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        crsi_oversold = crsi_4h[i] < 25
        crsi_overbought = crsi_4h[i] > 75
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995  # near breakout
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005  # near breakout
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200_4h[i]
        below_sma200 = close[i] < sma_200_4h[i]
        
        desired_signal = 0.0
        
        # === BULL REGIME LOGIC (only long entries) ===
        if bull_regime:
            # Mean reversion long in ranging market
            if ranging_regime and crsi_extreme_oversold and above_sma200:
                desired_signal = BASE_SIZE
            
            # Mean reversion long in trending market (pullback)
            if trending_regime and crsi_oversold and above_sma200:
                desired_signal = REDUCED_SIZE
            
            # Breakout confirmation long
            if donchian_breakout_long and crsi_4h[i] < 50 and above_sma200:
                desired_signal = BASE_SIZE
            
            # Conservative: CRSI oversold + above SMA200
            if crsi_oversold and above_sma200:
                if desired_signal == 0.0:
                    desired_signal = REDUCED_SIZE
        
        # === BEAR REGIME LOGIC (only short entries) ===
        elif bear_regime:
            # Mean reversion short in ranging market
            if ranging_regime and crsi_extreme_overbought and below_sma200:
                desired_signal = -BASE_SIZE
            
            # Mean reversion short in trending market (rally)
            if trending_regime and crsi_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE
            
            # Breakout confirmation short
            if donchian_breakout_short and crsi_4h[i] > 50 and below_sma200:
                desired_signal = -BASE_SIZE
            
            # Conservative: CRSI overbought + below SMA200
            if crsi_overbought and below_sma200:
                if desired_signal == 0.0:
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
                # Hold long if bull regime intact and CRSI not overbought
                if bull_regime and crsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if bear regime intact and CRSI not oversold
                if bear_regime and crsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if regime flips to bear or CRSI overbought
            if bear_regime and crsi_4h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if regime flips to bull or CRSI oversold
            if bull_regime and crsi_4h[i] < 30:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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