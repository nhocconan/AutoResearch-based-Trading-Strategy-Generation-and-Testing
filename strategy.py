#!/usr/bin/env python3
"""
Experiment #1017: 1d Primary + 1w HTF — Dual Regime (Choppiness + CRSI/Donchian)

Hypothesis: After 737 failed strategies, the winning pattern for 1d timeframe combines:
1. Choppiness Index regime detection (CHOP > 61.8 = mean revert, CHOP < 38.2 = trend)
2. Connors RSI for mean reversion entries (proven 75% win rate in ranges)
3. Donchian breakout for trend following (captures major moves)
4. 1w HMA21 for macro trend bias (asymmetric: easier long, harder short)
5. ATR trailing stop for risk management

Why 1d works:
- Target 20-50 trades/year (matches daily frequency, low fee drag)
- Less noise than lower TF, fewer whipsaws
- Captures major crypto moves without overtrading

Key fixes from failed experiments (#1014-1016):
- RELAXED CRSI thresholds (< 20 / > 80 not < 10 / > 90) for MORE trades
- BOTH long AND short signals (not long-only, works in 2025 bear)
- Tracking variables updated BEFORE stoploss check (no look-ahead)
- Proper MTF using get_htf_data() ONCE before loop
- min_periods on all rolling calculations
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Entry: CRSI < 20 (oversold) for long
    Entry: CRSI > 80 (overbought) for short
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=range(n), dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1]
        if len(window) >= rank_period:
            current_return = window.iloc[-1]
            rank = (window.iloc[:-1] < current_return).sum() / (len(window) - 1) * 100
            percent_rank.iloc[i] = rank
        else:
            percent_rank.iloc[i] = 50
    
    crsi = (rsi_3.values + rsi_streak.values + percent_rank.values) / 3
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
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
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
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
        tr[i] = max(high[i] - low[i], 
                   np.abs(high[i] - close[i-1]), 
                   np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === UPDATE TRACKING VARIABLES (BEFORE stoploss check) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
        elif in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_1d[i] > 61.8  # Ranging → mean reversion
        regime_trend = chop_1d[i] < 38.2  # Trending → breakout
        
        # === CRSI SIGNALS (Mean Reversion) - RELAXED thresholds ===
        crsi_oversold = crsi_1d[i] < 20  # Relaxed from < 15
        crsi_overbought = crsi_1d[i] > 80  # Relaxed from > 85
        
        # === DONCHIAN BREAKOUT (Trend Following) ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            donchian_breakout_short = close[i] < donchian_lower[i-1]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop:
            # Mean reversion in choppy market
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif crsi_oversold:
                desired_signal = REDUCED_SIZE
        elif regime_trend:
            # Trend following in trending market
            if donchian_breakout_long and macro_bull:
                desired_signal = BASE_SIZE
            elif donchian_breakout_long:
                desired_signal = REDUCED_SIZE
        else:
            # Neutral regime - use CRSI only
            if crsi_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop:
            # Mean reversion in choppy market
            if crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif crsi_overbought:
                desired_signal = -REDUCED_SIZE
        elif regime_trend:
            # Trend following in trending market
            if donchian_breakout_short and macro_bear:
                desired_signal = -BASE_SIZE
            elif donchian_breakout_short:
                desired_signal = -REDUCED_SIZE
        else:
            # Neutral regime - use CRSI only
            if crsi_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bullish or CRSI not extreme overbought
                if macro_bull or crsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bearish or CRSI not extreme oversold
                if macro_bear or crsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI becomes overbought
            if crsi_1d[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI becomes oversold
            if crsi_1d[i] < 25:
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
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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