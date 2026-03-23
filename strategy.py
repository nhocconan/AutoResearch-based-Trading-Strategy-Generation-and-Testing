#!/usr/bin/env python3
"""
Experiment #088: 30m Primary + 4h/1d HTF — Connors RSI Pullback with Choppiness Regime

Hypothesis: 30m strategies fail due to too many trades (fee drag) OR too few trades (0 signals).
This version uses PROVEN pattern from research: Connors RSI for entry timing + HTF trend filter.

Key design:
1) 1d HMA = macro bias (only long if price > 1d HMA, only short if price < 1d HMA)
2) 4h HMA = intermediate trend (confirms direction)
3) 30m Connors RSI = entry trigger (CRSI < 15 for long, CRSI > 85 for short)
4) Choppiness Index = regime filter (CHOP > 55 = range → use mean reversion, CHOP < 45 = trend → use pullback)
5) Session filter = only 8-20 UTC (highest volume, avoid Asian chop)
6) Volume filter = must be > 0.8x 20-period average
7) ATR stoploss = 2.0x trailing

Why this should work on 30m:
- HTF filters (1d/4h) ensure we only trade WITH the macro trend
- Connors RSI is more selective than regular RSI (75% win rate in literature)
- Session + volume filters reduce trades to 30-80/year target
- Discrete sizing (0.20/0.30) minimizes fee churn

Position size: 0.20 base, 0.30 max with confluence
Stoploss: 2.0*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_session_4h1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of recent closes lower than current close
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, period_rsi)
    
    # RSI Streak - measure consecutive up/down moves
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak magnitude to 0-100 scale
            streak_rsi[i] = min(100.0, max(0.0, 50.0 + streak[i] * 10.0))
    
    # Percent Rank - what % of last N closes are lower than current
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = close[i-period_rank+1:i+1]
        count_lower = np.sum(window[:-1] < window[-1])
        percent_rank[i] = (count_lower / (period_rank - 1)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_3_2_100 = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi_3_2_100[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(ema_21[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20  # UTC 8-20 (London + NY overlap)
        
        # Volume filter
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === HTF TREND BIAS (1d + 4h HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0  # trending market
        chop_ranging = chop_14[i] > 55.0  # ranging market
        
        # === CONNORS RSI EXTREMES (very selective) ===
        crsi_oversold = crsi_3_2_100[i] < 15.0  # extreme oversold
        crsi_overbought = crsi_3_2_100[i] > 85.0  # extreme overbought
        crsi_neutral_long = crsi_3_2_100[i] < 40.0  # not overbought for long
        crsi_neutral_short = crsi_3_2_100[i] > 60.0  # not oversold for short
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d uptrend + 4h uptrend + CRSI oversold + session + volume ---
        # Must have: 1d HMA bullish + 4h HMA bullish + CRSI extreme + session + volume
        if price_above_hma_1d and price_above_hma_4h and in_session and volume_ok:
            # Primary entry: CRSI extreme oversold
            if crsi_oversold:
                new_signal = POSITION_SIZE_MAX
            # Secondary entry: CRSI neutral + trending regime + EMA confirmation
            elif crsi_neutral_long and chop_trending and ema_bullish:
                new_signal = POSITION_SIZE_BASE
            # Range market: mean reversion at lower bound
            elif crsi_neutral_long and chop_ranging:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY: 1d downtrend + 4h downtrend + CRSI overbought + session + volume ---
        if price_below_hma_1d and price_below_hma_4h and in_session and volume_ok:
            # Primary entry: CRSI extreme overbought
            if crsi_overbought:
                new_signal = -POSITION_SIZE_MAX
            # Secondary entry: CRSI neutral + trending regime + EMA confirmation
            elif crsi_neutral_short and chop_trending and ema_bearish:
                new_signal = -POSITION_SIZE_BASE
            # Range market: mean reversion at upper bound
            elif crsi_neutral_short and chop_ranging:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Keep position if CRSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi_3_2_100[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi_3_2_100[i] > 20.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d or 4h HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d or price_below_hma_4h:
                new_signal = 0.0
        
        # Exit short if 1d or 4h HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d or price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi_3_2_100[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_3_2_100[i] < 20.0:
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