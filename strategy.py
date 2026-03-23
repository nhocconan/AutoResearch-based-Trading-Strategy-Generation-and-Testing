#!/usr/bin/env python3
"""
Experiment #628: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to excessive trades → fee drag.
This strategy uses 1d/4h HMA for TREND DIRECTION (HTF frequency) and 30m only for
ENTRY TIMING within that trend. Key innovations:
1. Choppiness Index regime filter: CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
2. Connors RSI (CRSI) for precise pullback entries: (RSI3 + RSI_Streak2 + PercentRank100) / 3
3. Session filter: Only trade 8-20 UTC (high liquidity, avoid Asian chop)
4. Volume filter: Only enter when volume > 0.8x 20-bar average
5. 1d HMA slope for primary bias, 4h HMA for confirmation
6. Conservative sizing (0.22) for lower TF to minimize fee impact

Why this might beat Sharpe=0.520:
- HTF trend filter reduces whipsaws (1d/4h HMA slope)
- CRSI catches extreme pullbacks with 75%+ win rate in backtests
- Choppiness regime adapts to market state (trend vs range)
- Session filter cuts 50%+ of low-quality trades
- Volume confirmation ensures real moves, not fakeouts
- Target: 40-70 trades/year (per Rule 10 for 30m)

Position sizing: 0.22 discrete (smaller for 30m per Rule 4)
Stoploss: 2.0*ATR trailing
Target trades: 40-70/year on 30m (strict confluence = fewer trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
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
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(close, 100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentile of current close vs last 100 closes
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_pos = np.where(streak > 0, streak, 0)
    streak_neg = np.where(streak < 0, -streak, 0)
    
    streak_gain = pd.Series(streak_pos).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss = pd.Series(streak_neg).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = streak_gain / (streak_loss + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // 1000 // 3600) % 24
    return hour

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
    
    # Calculate 1d HMA for primary trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h HMA for confirmation
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Session hours (UTC): 8-20 (high liquidity period)
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    in_session = (utc_hours >= 8) & (utc_hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D PRIMARY TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H CONFIRMATION (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 55.0  # Range market → mean revert
        chop_trend = chop_14[i] < 45.0  # Trending market → follow trend
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong buy signal
        crsi_overbought = crsi[i] > 85.0  # Strong sell signal
        crsi_neutral = 35.0 <= crsi[i] <= 65.0  # Pullback zone
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] > 0.8  # At least 80% of average volume
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull + 4h confirm + CRSI oversold/pullback + session + volume ---
        # Regime 1: Trending (CHOP < 45) → follow trend on pullback
        # Regime 2: Range (CHOP > 55) → mean revert at extremes
        if hma_1d_slope_bull and price_above_hma_1d:
            if hma_4h_slope_bull:  # 4h confirms 1d trend
                if chop_trend:
                    # Trending: enter on CRSI pullback (35-65)
                    if crsi_neutral and session_ok and volume_ok:
                        new_signal = POSITION_SIZE
                elif chop_range:
                    # Range: enter on CRSI oversold (<15)
                    if crsi_oversold and session_ok and volume_ok:
                        new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear + 4h confirm + CRSI overbought/pullback + session + volume ---
        elif hma_1d_slope_bear and price_below_hma_1d:
            if hma_4h_slope_bear:  # 4h confirms 1d trend
                if chop_trend:
                    # Trending: enter on CRSI pullback (35-65)
                    if crsi_neutral and session_ok and volume_ok:
                        new_signal = -POSITION_SIZE
                elif chop_range:
                    # Range: enter on CRSI overbought (>85)
                    if crsi_overbought and session_ok and volume_ok:
                        new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
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