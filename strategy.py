#!/usr/bin/env python3
"""
Experiment #488: 30m Primary + 4h/1d HTF — Simplified Regime + CRSI Mean Reversion

Hypothesis: After 477 failed experiments, clear pattern emerges for lower TF:
1. 30m strategies FAIL when entry conditions are too strict (see #478, #480, #486 = 0 trades)
2. OR logic for entries (any condition triggers) beats AND logic (all must agree)
3. HTF (4h) for trend direction + 30m for entry timing = proven pattern
4. CRSI thresholds must be relaxed (25/75 not 10/90) to get adequate frequency
5. Session filter may be killing trades — remove or make permissive
6. Volume filter should be >0.5x avg (permissive) not >0.8x (restrictive)

Why this might beat current best (Sharpe=0.435):
- Simpler entry logic = more trades (critical: need >=30 trades/symbol on train)
- 4h HMA provides clean trend bias without over-filtering
- CRSI(3,2,100) catches mean reversion better than standard RSI in range markets
- Choppiness Index regime detection switches between trend/mean-revert modes
- ATR 2.5x trailing stop protects in crashes while allowing room to breathe
- Discrete sizing (0.25 long, 0.20 short) minimizes fee churn

Position sizing: 0.20-0.25 (smaller for 30m TF, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test

CRITICAL: Use OR logic for entries, not AND. Multiple pathways to trigger.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_chop_4h1d_simp_v3"
timeframe = "30m"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_4h_21_aligned[i]
        bear_regime = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA crossover confirmation
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_30m[i] > 55.0
        is_trending = chop_30m[i] < 45.0
        
        # === CONNORS RSI SIGNALS (relaxed thresholds for frequency) ===
        crsi_oversold = crsi_30m[i] < 30.0
        crsi_overbought = crsi_30m[i] > 70.0
        crsi_extreme_oversold = crsi_30m[i] < 20.0
        crsi_extreme_overbought = crsi_30m[i] > 80.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME FILTER (permissive) ===
        volume_ok = vol_ratio[i] > 0.5
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — OR CONDITIONS (ANY can trigger) ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple pathways - OR logic)
        long_conditions = [
            # Path 1: Bull regime + CRSI oversold
            (bull_regime and crsi_oversold and volume_ok),
            # Path 2: Extreme CRSI oversold (strong signal)
            (crsi_extreme_oversold and volume_ok),
            # Path 3: Ranging market + mean reversion
            (is_ranging and crsi_30m[i] < 25.0 and above_sma200),
            # Path 4: Trending + pullback
            (is_trending and bull_regime and crsi_30m[i] < 35.0),
            # Path 5: RSI + CRSI confluence
            (rsi_oversold and crsi_30m[i] < 35.0 and above_sma200),
            # Path 6: 4h HMA bullish + CRSI dip
            (hma_4h_bullish and crsi_30m[i] < 40.0 and volume_ok),
        ]
        
        if any(long_conditions):
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (multiple pathways - OR logic)
        if new_signal == 0.0:
            short_conditions = [
                # Path 1: Bear regime + CRSI overbought
                (bear_regime and crsi_overbought and volume_ok),
                # Path 2: Extreme CRSI overbought (strong signal)
                (crsi_extreme_overbought and volume_ok),
                # Path 3: Ranging market + mean reversion
                (is_ranging and crsi_30m[i] > 75.0 and below_sma200),
                # Path 4: Trending + pullback
                (is_trending and bear_regime and crsi_30m[i] > 65.0),
                # Path 5: RSI + CRSI confluence
                (rsi_overbought and crsi_30m[i] > 65.0 and below_sma200),
                # Path 6: 4h HMA bearish + CRSI rally
                (hma_4h_bearish and crsi_30m[i] > 60.0 and volume_ok),
            ]
            
            if any(short_conditions):
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long when CRSI overbought
        if in_position and position_side > 0 and crsi_30m[i] > 75.0:
            new_signal = 0.0
        # Exit short when CRSI oversold
        if in_position and position_side < 0 and crsi_30m[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (4h trend reversal)
        if in_position and position_side > 0 and bear_regime and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime and hma_4h_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals