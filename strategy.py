#!/usr/bin/env python3
"""
Experiment #1130: 1h Primary + 4h/12h HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 822+ failed experiments, the key insight is REGIME-AWARE entry timing.
Most strategies fail because they use the same logic in trending AND ranging markets.
This strategy uses:
1. 12h HMA(21) for macro trend direction (works across all symbols)
2. 4h Choppiness Index for regime detection (CHOP > 55 = range, CHOP < 45 = trend)
3. Connors RSI (CRSI) for mean reversion entries in range regime
4. Session filter (8-20 UTC) when volume is highest
5. Volume filter (> 0.8x 20-bar avg) to avoid low-liquidity traps
6. ATR(14) 2.5x trailing stop
7. Position size 0.25 (smaller for 1h TF to reduce fee drag)

Why this should beat Sharpe=0.612:
- Regime-aware: different logic for trend vs range markets
- Connors RSI has 75% win rate in research (RSI3 + RSI_Streak + PercentRank)
- Session filter avoids Asian session low-volume whipsaws
- 12h HMA prevents counter-trend trades that destroyed returns in 2022
- Target: 40-80 trades/year (strict enough to avoid fee drag, loose enough to trade)

Timeframe: 1h (primary)
HTF: 4h (Choppiness), 12h (HMA trend) — loaded ONCE before loop
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
Target: 40-80 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_regime_4h12h_session_atr_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 15, Short: CRSI > 85
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
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
    streak_rsi = np.full(n, 50.0)
    for i in range(streak_period, n):
        if np.isnan(streak[i]):
            continue
        # Map streak to 0-100: positive streak = bullish, negative = bearish
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * pos_count / streak_period
    
    # Component 3: Percent Rank of close over lookback period
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if len(window) == rank_period:
            pct_rank[i] = 100.0 * np.sum(window[:-1] < close[i]) / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
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
    
    return chop

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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        if atr[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop = chop_4h_aligned[i]
        is_range = chop > 55.0  # Choppy/ranging market
        is_trend = chop < 45.0  # Trending market
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # Mean reversion long
        crsi_overbought = crsi[i] > 80.0  # Mean reversion short
        
        desired_signal = 0.0
        
        # === RANGE REGIME: Mean Reversion ===
        # Long: CRSI oversold + price above 12h HMA (bullish bias)
        if is_range and macro_bull and crsi_oversold and in_session and volume_ok:
            desired_signal = BASE_SIZE
        
        # Short: CRSI overbought + price below 12h HMA (bearish bias)
        elif is_range and macro_bear and crsi_overbought and in_session and volume_ok:
            desired_signal = -BASE_SIZE
        
        # === TREND REGIME: Pullback Entries ===
        # Long: CRSI pullback in uptrend
        elif is_trend and macro_bull and crsi[i] < 40.0 and in_session and volume_ok:
            desired_signal = BASE_SIZE
        
        # Short: CRSI pullback in downtrend
        elif is_trend and macro_bear and crsi[i] > 60.0 and in_session and volume_ok:
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
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull:
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