#!/usr/bin/env python3
"""
Experiment #010: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + Session Filter

Hypothesis: Previous strategies failed due to either too many trades (fee drag) or too few (0 trades).
This combines proven elements differently:
1. Connors RSI (CRSI) for mean reversion entries — research shows 75% win rate at extremes
2. Choppiness Index (CHOP) for regime detection — avoid trend-following in ranges
3. 12h HMA for HTF trend bias — asymmetric entries only
4. Session filter (8-20 UTC) — only trade during high liquidity
5. Volume confirmation — avoid low-volume fakeouts

Key differences from failed attempts:
- NOT using regular RSI (failed in #007, #009) — using Connors RSI instead
- NOT using Donchian breakouts (failed in #002, #009) — using CRSI extremes
- NOT using pure vol spike (failed in #004) — using CHOP regime filter
- 1h TF with VERY strict confluence (3+ filters) to limit trades to 30-60/year

Why this might work:
- CRSI at <10 or >90 is rare (~5% of bars) = natural trade limiter
- CHOP filter prevents entering mean-reversion during strong trends
- Session filter cuts overnight low-liquidity traps
- 12h HMA ensures we don't fight the macro trend
- Position size 0.25 (conservative for 1h per Rule 4)

Entry conditions (strict but NOT impossible):
- Long: CRSI < 12 AND (CHOP > 50 OR 12h HMA bullish) AND session 8-20 UTC AND volume > 0.7x avg
- Short: CRSI > 88 AND (CHOP > 50 OR 12h HMA bearish) AND session 8-20 UTC AND volume > 0.7x avg

Stoploss: 2.5*ATR trailing, signal→0 when hit
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_session_12h_v1"
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
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Research shows CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, period_rsi)
    
    # Component 2: RSI on streak duration
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on absolute streak values (convert to positive for RSI calc)
    streak_positive = np.abs(streak) + 1e-10
    rsi_streak = calculate_rsi(streak_positive, period_streak)
    
    # Component 3: PercentRank of close over last 100 bars
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = close[i-period_rank:i]
        rank = np.sum(window < close[i])
        percent_rank[i] = rank / period_rank * 100.0
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low + 1e-10
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    return dt.hour

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
    
    # Calculate 12h HMA for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h Choppiness Index for regime detection
    chop_4h = calculate_choppiness_index(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    
    # Volume SMA for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(volume_sma[i]):
            continue
        if atr_14[i] == 0 or volume_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === 12H TREND BIAS ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-5] if i >= 5 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-5] if i >= 5 else False
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        chop_value = chop_4h_aligned[i]
        is_ranging = chop_value > 50  # Range market = mean reversion OK
        is_trending = chop_value < 45  # Trend market = follow trend
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 12  # Research: <10 is extreme, using <12 for more trades
        crsi_overbought = crsi[i] > 88  # Research: >90 is extreme, using >88 for more trades
        
        # === ASYMMETRIC ENTRY LOGIC (3+ confluence required) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Need: CRSI oversold + (ranging OR bullish trend) + session + volume
        long_condition_1 = crsi_oversold  # Mean reversion signal
        long_condition_2 = is_ranging or (hma_12h_slope_bull and price_above_hma_12h)  # Regime OK
        long_condition_3 = in_session  # Liquidity
        long_condition_4 = volume_ok  # Volume confirmation
        
        if long_condition_1 and long_condition_2 and long_condition_3 and long_condition_4:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Need: CRSI overbought + (ranging OR bearish trend) + session + volume
        short_condition_1 = crsi_overbought  # Mean reversion signal
        short_condition_2 = is_ranging or (hma_12h_slope_bear and price_below_hma_12h)  # Regime OK
        short_condition_3 = in_session  # Liquidity
        short_condition_4 = volume_ok  # Volume confirmation
        
        if short_condition_1 and short_condition_2 and short_condition_3 and short_condition_4:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
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