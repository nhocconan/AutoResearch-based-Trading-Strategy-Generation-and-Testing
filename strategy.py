#!/usr/bin/env python3
"""
Experiment #480: 1h Primary + 4h/12h HTF — Regime-Adaptive CRSI Mean Reversion

Hypothesis: After 479 experiments, clear patterns emerge for lower TF success:
1. 1h timeframe needs VERY strict filters (target 40-60 trades/year max)
2. HTF (4h/12h) MUST dictate direction, 1h only for entry timing
3. Previous 1h failures (#470, #475, #478) had Sharpe=0.000 = 0 trades
4. Too many conflicting filters = no trades. Need OR logic for entries.
5. Connors RSI extremes work best for mean reversion in bear/range markets
6. Session filter (8-20 UTC) reduces false signals during low liquidity

Why this might beat current best (Sharpe=0.435):
- 4h HMA provides cleaner trend bias than 1d for 1h entries
- 12h Choppiness regime detection adapts to market conditions
- Relaxed CRSI thresholds (20/80 instead of 10/90) for adequate frequency
- Volume + session filters reduce noise without killing trade count
- Asymmetric sizing (0.25 long, 0.20 short) protects in bear markets
- 2.5x ATR stoploss accounts for 1h noise while protecting capital

Position sizing: 0.20-0.25 (smaller for lower TF, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-60 trades/year on 1h, >=160 trades/symbol on train, >=40 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_4h12h_session_v1"
timeframe = "1h"
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
    
    Proven 75% win rate for mean reversion entries.
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
    
    # Component 3: Percent Rank of returns over 100 periods
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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 12h HTF indicators (regime detection)
    chop_12h = calculate_choppiness(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, period=200)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(chop_12h_aligned[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # Extract UTC hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20  # High liquidity hours
        
        # Volume filter (not too strict)
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === 4H MAJOR TREND (primary direction filter) ===
        bull_trend = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        bear_trend = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Price vs 4h HMA confirmation
        price_above_hma = close[i] > hma_4h_21_aligned[i]
        price_below_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H CHOPPINESS REGIME ===
        is_ranging = chop_12h_aligned[i] > 55.0
        is_trending = chop_12h_aligned[i] < 45.0
        
        # === CONNORS RSI SIGNALS (relaxed for frequency) ===
        crsi_oversold = crsi_1h[i] < 25.0
        crsi_overbought = crsi_1h[i] > 75.0
        crsi_extreme_oversold = crsi_1h[i] < 15.0
        crsi_extreme_overbought = crsi_1h[i] > 85.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — OR CONDITIONS FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (any condition triggers - OR logic for frequency)
        if in_session and volume_ok:
            # Condition 1: Bull trend + CRSI oversold
            if bull_trend and crsi_oversold:
                new_signal = LONG_SIZE
            # Condition 2: Price above HMA + extreme oversold
            elif price_above_hma and crsi_extreme_oversold:
                new_signal = LONG_SIZE
            # Condition 3: Ranging market + oversold + above SMA200
            elif is_ranging and crsi_oversold and above_sma200:
                new_signal = LONG_SIZE * 0.8
            # Condition 4: Trending market + bull trend + mild oversold
            elif is_trending and bull_trend and crsi_1h[i] < 35.0:
                new_signal = LONG_SIZE
            # Condition 5: Extreme oversold alone (catch deep dips)
            elif crsi_extreme_oversold and above_sma200:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (any condition triggers)
        if new_signal == 0.0 and in_session and volume_ok:
            # Condition 1: Bear trend + CRSI overbought
            if bear_trend and crsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Price below HMA + extreme overbought
            elif price_below_hma and crsi_extreme_overbought:
                new_signal = -SHORT_SIZE
            # Condition 3: Ranging market + overbought + below SMA200
            elif is_ranging and crsi_overbought and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 4: Trending market + bear trend + mild overbought
            elif is_trending and bear_trend and crsi_1h[i] > 65.0:
                new_signal = -SHORT_SIZE
            # Condition 5: Extreme overbought alone (catch rallies)
            elif crsi_extreme_overbought and below_sma200:
                new_signal = -SHORT_SIZE * 0.7
        
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
        if in_position and position_side > 0 and crsi_1h[i] > 80.0:
            new_signal = 0.0
        # Exit short when CRSI oversold
        if in_position and position_side < 0 and crsi_1h[i] < 20.0:
            new_signal = 0.0
        
        # Regime flip exit (trend reversal)
        if in_position and position_side > 0 and bear_trend and price_below_hma:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_trend and price_above_hma:
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            # If same side, keep tracking highest/lowest
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                if lowest_since_entry == 0.0:
                    lowest_since_entry = close[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals