#!/usr/bin/env python3
"""
Experiment #475: 15m Extreme Mean Reversion + 4h HMA Bias + CHOP Regime + ATR Stop
Hypothesis: 15m trend following fails due to noise/fees. Instead, use extreme mean
reversion (CRSI <10 or >90) ONLY when 4h trend confirms direction. Choppiness Index
filters out ambiguous regimes. This produces fewer but higher quality trades with
better win rate. Focus on capturing snapback moves in range markets (2025 bear/range).
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_extreme_4h_hma_chop_regime_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market (favor mean reversion)
    CHOP < 38.2 = trending market (favor trend following)
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    mask = (range_hl > 0) & (atr_sum > 0)
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Extreme values <10 or >90 signal strong mean reversion opportunities.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI on close (period=3)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streaks (consecutive up/down days)
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak absolute values (period=2)
    streak_abs = np.abs(streak) + 0.001  # avoid zero
    rsi_streak = calculate_rsi(streak_abs, streak_period)
    
    # Percent rank of price change (vectorized)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window_changes = np.diff(close[i-rank_period:i+1])
        if len(window_changes) > 0:
            current_change = delta[i]
            pct_rank[i] = np.sum(window_changes <= current_change) / len(window_changes) * 100
    
    # Combine into CRSI
    start_idx = max(rsi_period, streak_period, rank_period)
    for i in range(start_idx, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pct_rank[i]) / 3
    
    return crsi

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_15m = calculate_hma(close, 21)
    chop = calculate_choppiness(high, low, close, 14)
    crsi = calculate_crsi(close)
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - MUST align with entry direction
        four_h_bullish = close[i] > hma_4h_aligned[i]
        four_h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m mean reversion context
        price_below_hma = close[i] < hma_15m[i]
        price_above_hma = close[i] > hma_15m[i]
        
        # Regime detection (CHOP)
        is_range = chop[i] > 55  # Range market - favor mean reversion
        is_trend = chop[i] < 45  # Trend market - favor trend continuation
        
        # CRSI extremes (mean reversion signals)
        crsi_extreme_low = crsi[i] < 15  # Oversold
        crsi_extreme_high = crsi[i] > 85  # Overbought
        
        # Z-score extremes (additional confirmation)
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Range market + CRSI extreme low + 4h bullish bias
        if is_range and crsi_extreme_low and four_h_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: Range market + CRSI low + Z-score low + 4h bullish
        elif is_range and crsi[i] < 25 and zscore_extreme_low and four_h_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Trend market + Price below HMA + CRSI low (pullback in uptrend)
        elif is_trend and four_h_bullish and price_below_hma and crsi[i] < 30:
            new_signal = SIZE_ENTRY
        # Path 4: CRSI very extreme (<10) regardless of regime + 4h bullish
        elif crsi[i] < 10 and four_h_bullish:
            new_signal = SIZE_ENTRY
        # Path 5: Z-score very extreme (<-2.0) + 4h bullish (deep mean reversion)
        elif zscore[i] < -2.0 and four_h_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Range market + CRSI extreme high + 4h bearish bias
        if is_range and crsi_extreme_high and four_h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: Range market + CRSI high + Z-score high + 4h bearish
        elif is_range and crsi[i] > 75 and zscore_extreme_high and four_h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Trend market + Price above HMA + CRSI high (rally in downtrend)
        elif is_trend and four_h_bearish and price_above_hma and crsi[i] > 70:
            new_signal = -SIZE_ENTRY
        # Path 4: CRSI very extreme (>90) regardless of regime + 4h bearish
        elif crsi[i] > 90 and four_h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 5: Z-score very extreme (>2.0) + 4h bearish (deep mean reversion)
        elif zscore[i] > 2.0 and four_h_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (1.5*ATR for 15m - tighter than higher TFs)
            current_stop = highest_close - 1.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 1.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (1.5*ATR for 15m)
            current_stop = lowest_close + 1.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 1.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 1.5 * atr[i] if position_side > 0 else close[i] + 1.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 1.5 * atr[i] if position_side > 0 else close[i] + 1.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals