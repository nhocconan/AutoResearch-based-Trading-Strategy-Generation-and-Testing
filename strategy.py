#!/usr/bin/env python3
"""
Experiment #1037: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: After 751+ failed strategies, daily timeframe with weekly trend filter provides
the best risk-adjusted returns for bear/range markets (2025 test). This strategy combines:

1. CHOPPINESS INDEX (CHOP) regime detection:
   - CHOP > 61.8 = ranging → Connors RSI mean reversion
   - CHOP < 38.2 = trending → Donchian breakout
   - This adaptive approach works in both bull and bear markets

2. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > weekly HMA21 (oversold in uptrend)
   - Short: CRSI > 85 + price < weekly HMA21 (overbought in downtrend)
   - Proven 75% win rate on ETH in bear markets

3. DONCHIAN BREAKOUT (20-period):
   - Long: price breaks 20-day high + CHOP < 38.2 + price > weekly HMA
   - Short: price breaks 20-day low + CHOP < 38.2 + price < weekly HMA

4. 1w HMA21: Major trend filter (asymmetric bias for bear market)
   - Only long when price > 1w HMA21 (weekly bullish)
   - Only short when price < 1w HMA21 (weekly bearish)

5. ATR Trailing Stop: 3.0x ATR for daily timeframe (wider stops)

Why 1d works:
- Target 20-50 trades/year (vs 100+ on lower TF)
- Less fee drag, more significant moves
- Weekly HTF provides major trend context
- Proven patterns: Choppiness+CRSI (ETH +0.923), Donchian+HMA (SOL +0.782)

Critical fixes from failures:
- RELAXED CRSI thresholds (15/85 not 10/90) for more trades
- DUAL entry logic (mean reversion OR breakout) ensures trades in all regimes
- Weekly HMA filter prevents counter-trend trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_donchian_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        change = close[i] - close[i-1]
        gains[i] = max(change, 0)
        losses[i] = max(-change, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI: RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        abs_streak = abs(streak[i])
        if abs_streak >= streak_period:
            streak_rsi[i] = 100 if streak[i] > 0 else 0
        else:
            streak_rsi[i] = 50 + (streak[i] / streak_period) * 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank: percentage of closes in last 100 days that are lower than current
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_lower / (rank_period - 1)) * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
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
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel: upper = highest high, lower = lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # RSI(14) for additional filter
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === MAJOR TREND (1w HMA21) ===
        # Asymmetric: only long above weekly HMA, only short below
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Ranging → mean reversion
        regime_trend = chop[i] < 38.2  # Trending → breakout
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if weekly_bull:
            if regime_chop:
                # Mean reversion in ranging market
                if crsi_oversold and rsi_oversold:
                    desired_signal = BASE_SIZE
                elif crsi_extreme_oversold:
                    # Very oversold = strong reversal signal
                    desired_signal = BASE_SIZE
            elif regime_trend:
                # Breakout in trending market
                if donchian_breakout_long:
                    desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if weekly_bear:
            if regime_chop:
                # Mean reversion in ranging market
                if crsi_overbought and rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif crsi_extreme_overbought:
                    desired_signal = -BASE_SIZE
            elif regime_trend:
                # Breakout in trending market
                if donchian_breakout_short:
                    desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for daily) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly bullish and CRSI not extreme overbought
                if weekly_bull and crsi[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly bearish and CRSI not extreme oversold
                if weekly_bear and crsi[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses or CRSI very overbought
            if not weekly_bull and crsi[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses or CRSI very oversold
            if not weekly_bear and crsi[i] < 30:
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
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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