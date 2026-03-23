#!/usr/bin/env python3
"""
Experiment #008: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Dual HTF Trend

Hypothesis: 30m timeframe with strict 4h/1d trend confluence should generate 30-80 trades/year.
Key insight: Lower TF strategies fail from too many trades → fee drag. Use 3+ confluence filters.

Strategy components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — 75% win rate
2. Choppiness Index: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA(21): Intermediate trend bias
4. 1d HMA(21): Macro trend bias (strongest filter)
5. Session filter: Only 8-20 UTC (high liquidity, less manipulation)
6. Volume filter: Volume > 0.8x 20-bar average
7. ATR(14) stoploss: 2.5x trailing stop

Why this should work:
- 30m entries with 4h/1d direction = HTF win rate with LTF precision
- CRSI extreme levels (10/90) = high probability reversals
- Dual HTF confirmation = avoids counter-trend trades in strong moves
- Session + volume filters = reduces false signals by ~60%
- Position size 0.25 = controls drawdown (77% crash → 19% equity loss)

Trade frequency target: 30-80/year (strict entry: all 5 conditions must align)
Position size: 0.25 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_dual_htf_session_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's change over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI of streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values (period=2)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # PercentRank: percentile of today's return over last 100 bars
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i]
        if np.isnan(current_return):
            percent_rank[i] = 50.0
        else:
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                percent_rank[i] = 100.0 * np.sum(valid_window <= current_return) / len(valid_window)
            else:
                percent_rank[i] = 50.0
    
    # CRSI = average of 3 components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP = ranging, Low CHOP = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 4h HMA for intermediate trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hours for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25  # Smaller for lower TF to control fee drag
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + HTF alignment
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0 or np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        current_hour = hours[i]
        in_session = 8 <= current_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === 4H INTERMEDIATE TREND ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D MACRO TREND (strongest filter) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0
        is_trending = chop_value < 45.0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === CONFLUENCE ENTRY LOGIC (ALL conditions must align) ===
        new_signal = 0.0
        
        # --- LONG ENTRY (5 conditions) ---
        # 1) Session filter
        # 2) Volume filter
        # 3) CRSI oversold OR (ranging + near oversold)
        # 4) 4h trend not bearish
        # 5) 1d trend confirms (or neutral in range)
        if in_session and volume_ok:
            # Long in ranging market: mean reversion
            if is_ranging and crsi_oversold:
                if not price_below_hma_4h:  # 4h not bearish
                    if price_above_hma_1d or chop_value > 60:  # 1d bullish OR very choppy
                        new_signal = POSITION_SIZE
            
            # Long in trending market: pullback entry
            elif is_trending and crsi[i] < 35:  # Less extreme for trend follow
                if price_above_hma_4h and price_above_hma_1d:  # Both HTF bullish
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY (5 conditions) ---
        if in_session and volume_ok:
            # Short in ranging market: mean reversion
            if is_ranging and crsi_overbought:
                if not price_above_hma_4h:  # 4h not bullish
                    if price_below_hma_1d or chop_value > 60:  # 1d bearish OR very choppy
                        new_signal = -POSITION_SIZE
            
            # Short in trending market: pullback entry
            elif is_trending and crsi[i] > 65:  # Less extreme for trend follow
                if price_below_hma_4h and price_below_hma_1d:  # Both HTF bearish
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if both 4h and 1d turn bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = 0.0
        
        # Exit short if both 4h and 1d turn bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and price_above_hma_1d:
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