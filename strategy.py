#!/usr/bin/env python3
"""
Experiment #495: 1h Primary + 4h/1d HTF — Strict Confluence + Session Filter

Hypothesis: Previous 1h failures (#485, #490) had Sharpe -1.996 and -2.553 due to:
1. Too many trades (>200/year) → fee drag destroyed profits
2. Insufficient HTF filtering (used 1h for everything)
3. Entry conditions too loose (2 confluence instead of 4+)

This version implements STRICT confluence for 1h:
- 4h HMA for TREND DIRECTION (not 1h)
- 1d HMA for MAJOR REGIME (bull/bear filter)
- Connors RSI <20/>80 for ENTRY TIMING (extreme only)
- Volume >1.2x 20-bar avg (confirm participation)
- Session filter 8-20 UTC (avoid Asian chop)
- Choppiness >55 = range (mean revert), <45 = trend (follow)

Why this might work when #485/#490 failed:
- 4x stricter entry (need ALL conditions, not ANY)
- Session filter cuts 60% of potential trades
- Volume filter avoids fake breakouts
- 4h trend + 1h entry = HTF frequency with LTF precision
- Target: 40-60 trades/year (not 200+)

Position sizing: 0.20-0.25 (smaller for 1h due to fee sensitivity)
Stoploss: 2.0 * ATR trailing
Leverage: 1.0 (no leverage)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_strict_confluence_session_4h1d_v1"
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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    utc_hour = pd.to_datetime(ts_seconds, unit='s').dt.hour
    return utc_hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_1h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    LONG_SIZE = 0.22
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
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(crsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoid Asian session chop (0-8 UTC) and late US (20-24 UTC)
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER (>1.2x average) ===
        volume_confirmed = volume[i] > 1.2 * vol_avg_20[i]
        
        # === 1D MAJOR REGIME (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION (signal direction) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === CONNORS RSI EXTREMES (strict thresholds) ===
        crsi_extreme_oversold = crsi_1h[i] < 20.0
        crsi_extreme_overbought = crsi_1h[i] > 80.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — STRICT CONFLUENCE (ALL conditions must match) ===
        new_signal = 0.0
        
        # LONG: Need 4h bullish + 1d bull regime + CRSI extreme + volume + session
        # In ranging market: mean revert long at extreme oversold
        if in_session and volume_confirmed:
            if is_ranging and crsi_extreme_oversold and above_sma200 and hma_4h_bullish:
                new_signal = LONG_SIZE
            # In trending market: only long if 4h trend + 1d regime + pullback
            elif is_trending and bull_regime_1d and hma_4h_bullish and crsi_1h[i] < 35.0:
                new_signal = LONG_SIZE
            # Extreme CRSI override (very rare, high conviction)
            elif crsi_1h[i] < 15.0 and above_sma200 and hma_4h_bullish:
                new_signal = LONG_SIZE
        
        # SHORT: Need 4h bearish + 1d bear regime + CRSI extreme + volume + session
        if new_signal == 0.0 and in_session and volume_confirmed:
            if is_ranging and crsi_extreme_overbought and below_sma200 and hma_4h_bearish:
                new_signal = -SHORT_SIZE
            elif is_trending and bear_regime_1d and hma_4h_bearish and crsi_1h[i] > 65.0:
                new_signal = -SHORT_SIZE
            elif crsi_1h[i] > 85.0 and below_sma200 and hma_4h_bearish:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long when CRSI overbought
        if in_position and position_side > 0 and crsi_1h[i] > 75.0:
            new_signal = 0.0
        # Exit short when CRSI oversold
        if in_position and position_side < 0 and crsi_1h[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (4h trend reversal)
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish:
            new_signal = 0.0
        
        # 1d regime flip exit (major trend change)
        if in_position and position_side > 0 and bear_regime_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1d:
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