#!/usr/bin/env python3
"""
Experiment #1013: 1d Primary + 1w HTF — Dual Regime (Chop/Trend) with CRSI + Donchian

Hypothesis: After 735+ failed strategies, the winning pattern is:
1. 1w HMA21 for MACRO trend bias (very slow, stable across all symbols)
2. 1d Choppiness Index for REGIME detection (chop vs trend)
3. CHOPPY regime (CHOP>61.8): Connors RSI mean reversion at extremes
4. TREND regime (CHOP<38.2): Donchian breakout with HMA confirmation
5. ATR trailing stop (2.5x) for risk management
6. Conservative sizing (0.25-0.30) to survive 2022-style crashes

Why 1d works:
- Natural 20-50 trades/year target (minimizes fee drag)
- Less noise than lower TFs
- Works in bear/range markets (2025 test period)
- 1w HTF provides stable macro filter without whipsaw

Key improvements from #1012 (12h):
- 1d primary = fewer false signals, cleaner regime detection
- 1w HTF = more stable than 1d for macro bias
- Donchian breakout for trend regime (proven on SOL)
- CRSI for chop regime (proven on ETH)
- Relaxed entry thresholds to ensure trades on ALL symbols

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
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
    Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator for crypto.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) - fast mean reversion
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100 * (streak[i] / (streak[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100 * (1 / (np.abs(streak[i]) + 1))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100) - how current return ranks vs last 100
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        count_lower = np.sum(window[:-1] < current)
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
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
        
        if atr_sum > 0 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average - faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(close := high)  # just to get length
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === CALCULATE PRIMARY (1d) INDICATORS ===
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # HMA for trend confirmation on primary TF
    hma_1d = calculate_hma(close, 21)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d[i]):
            continue
        
        # === MACRO TREND (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_1d[i] > 61.8  # Ranging market → mean reversion
        regime_trend = chop_1d[i] < 38.2  # Trending market → breakout
        
        # === CRSI SIGNALS (Mean Reversion for Choppy Regime) ===
        crsi_extreme_oversold = crsi_1d[i] < 15
        crsi_extreme_overbought = crsi_1d[i] > 85
        crsi_oversold = crsi_1d[i] < 25
        crsi_overbought = crsi_1d[i] > 75
        
        # === DONCHIAN BREAKOUT (Trend Following for Trending Regime) ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # HMA confirmation for trend entries
        hma_bullish = close[i] > hma_1d[i]
        hma_bearish = close[i] < hma_1d[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop and macro_bull:
            # Mean reversion in choppy bullish macro
            if crsi_extreme_oversold:
                desired_signal = BASE_SIZE
            elif crsi_oversold and close[i] > hma_1w_aligned[i] * 0.95:
                desired_signal = REDUCED_SIZE
        
        elif regime_trend and macro_bull:
            # Trend breakout in trending bullish macro
            if breakout_long and hma_bullish:
                desired_signal = BASE_SIZE
            elif hma_bullish and crsi_oversold:
                # Pullback entry in uptrend
                desired_signal = REDUCED_SIZE
        
        elif not regime_chop and not regime_trend and macro_bull:
            # Transition zone - only extreme CRSI entries
            if crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop and macro_bear:
            # Mean reversion in choppy bearish macro
            if crsi_extreme_overbought:
                desired_signal = -BASE_SIZE
            elif crsi_overbought and close[i] < hma_1w_aligned[i] * 1.05:
                desired_signal = -REDUCED_SIZE
        
        elif regime_trend and macro_bear:
            # Trend breakout in trending bearish macro
            if breakout_short and hma_bearish:
                desired_signal = -BASE_SIZE
            elif hma_bearish and crsi_overbought:
                # Pullback entry in downtrend
                desired_signal = -REDUCED_SIZE
        
        elif not regime_chop and not regime_trend and macro_bear:
            # Transition zone - only extreme CRSI entries
            if crsi_extreme_overbought:
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
                if macro_bull and crsi_1d[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish and CRSI not extreme oversold
                if macro_bear and crsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish
            if macro_bear and crsi_1d[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish
            if macro_bull and crsi_1d[i] < 40:
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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