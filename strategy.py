#!/usr/bin/env python3
"""
Experiment #055: 1h Primary + 4h/1d HTF — CRSI Mean Reversion + CHOP Regime + Session Filter

Hypothesis: 1h timeframe with 4h trend bias + 1d regime filter + Connors RSI entries
will generate 30-80 trades/year with Sharpe > 0.486 by combining:
1) 4h HMA for macro trend direction (prevents counter-trend trades)
2) 1d Choppiness Index for regime detection (range vs trend mode)
3) Connors RSI (CRSI) for precise entry timing (proven 75% win rate)
4) Session filter (8-20 UTC) to avoid low-volume whipsaws
5) Volume confirmation (>0.8x 20-bar average)

Why this should work for 1h:
- 4h/1d HTF provides signal DIRECTION (not entry timing)
- 1h only used for ENTRY precision within HTF trend
- CRSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) for superior timing
- Session filter reduces trades by ~60% (only 12/24 hours)
- Volume filter ensures liquidity during entries
- Discrete position sizing (0.25) minimizes fee churn

Position size: 0.25 (conservative for 1h timeframe)
Stoploss: 2.5*ATR trailing
Target: 30-80 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_session_4h1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
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
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * (streak_abs[i] / (streak_abs[i] + 1))
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        percent_rank[i] = 100.0 * np.sum(window < close[i]) / rank_period
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
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
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
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
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d CHOP for regime detection
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D REGIME (Choppiness) ===
        chop_1d_value = chop_1d_aligned[i]
        is_ranging_1d = chop_1d_value > 55.0
        is_trending_1d = chop_1d_value < 45.0
        
        # === 1H CHOPPYNESS (local regime) ===
        chop_1h_value = chop_1h[i]
        is_ranging_1h = chop_1h_value > 55.0
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong long signal
        crsi_overbought = crsi[i] > 85.0  # Strong short signal
        crsi_neutral = 15.0 <= crsi[i] <= 85.0
        
        # === SMA200 FILTER ===
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours with volume confirmation
        if in_session and volume_ok:
            # --- TRENDING REGIME (1d CHOP < 45): Follow 4h trend with CRSI pullback ---
            if is_trending_1d:
                # Long: 4h uptrend + CRSI pullback oversold
                if price_above_hma_4h and crsi_oversold:
                    if price_above_sma_200:  # Additional confirmation
                        new_signal = POSITION_SIZE
                
                # Short: 4h downtrend + CRSI pullback overbought
                elif price_below_hma_4h and crsi_overbought:
                    if price_below_sma_200:  # Additional confirmation
                        new_signal = -POSITION_SIZE
            
            # --- RANGING REGIME (1d CHOP > 55): Mean reversion at extremes ---
            elif is_ranging_1d:
                # Long: CRSI very oversold + price above SMA200 (bullish bias)
                if crsi_oversold and price_above_sma_200:
                    new_signal = POSITION_SIZE
                
                # Short: CRSI very overbought + price below SMA200 (bearish bias)
                elif crsi_overbought and price_below_sma_200:
                    new_signal = -POSITION_SIZE
            
            # --- NEUTRAL REGIME: Use 1h CHOP + 4h trend ---
            else:
                # Long: 4h uptrend + 1h ranging + CRSI oversold
                if price_above_hma_4h and is_ranging_1h and crsi_oversold:
                    new_signal = POSITION_SIZE
                
                # Short: 4h downtrend + 1h ranging + CRSI overbought
                elif price_below_hma_4h and is_ranging_1h and crsi_overbought:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (avoid churn) ===
        if in_position and new_signal == 0.0:
            # Hold if CRSI not at opposite extreme
            if position_side > 0 and crsi[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 20.0:
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
        
        # === EXIT ON TREND CHANGE (4h HMA cross) ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
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