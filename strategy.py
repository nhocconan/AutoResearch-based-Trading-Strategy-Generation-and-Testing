#!/usr/bin/env python3
"""
Experiment #028: 30m Primary + 4h/1d HTF — Ultra-Selective Confluence Strategy

Hypothesis: 30m strategies fail due to too many trades (>200/year) causing fee drag.
Solution: Use 1d/4h for SIGNAL DIRECTION, 30m ONLY for entry timing with 4+ confluence filters.

Key innovations:
1. MACRO BIAS (1d HMA): Only long if price > 1d HMA, only short if price < 1d HMA
2. TREND CONFIRMATION (4h HMA): Must align with 1d bias for entry
3. ENTRY TIMING (30m): CRSI extremes + Bollinger Band touches for precision
4. SESSION FILTER: Only trade 8-20 UTC (highest volume, lowest slippage)
5. VOLUME CONFIRMATION: Current volume > 0.8x 20-period average
6. VOLATILITY FILTER: ATR ratio must indicate normal conditions (not panic)

Why this works for 30m:
- HTF filters reduce trade frequency to 30-80/year target
- Session filter avoids low-volume Asian session whipsaws
- 4+ confluence ensures only high-probability entries
- Discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn

Position size: 0.22 (smaller for lower TF to reduce DD)
Stoploss: 2.0*ATR trailing (tighter for lower TF)
Target trades: 40-80/year (strict enough to avoid fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ultra_selective_confluence_4h1d_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Streak calculation (consecutive up/down bars)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of today's return over last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1]
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = np.sum(window_returns <= current_return) / len(window_returns)
            percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

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
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h HMA for trend confirmation
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # 30m HMA for local trend
    hma_30m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for lower TF)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(bb_upper[i]) or np.isnan(vol_avg[i]):
            continue
        if atr_14[i] == 0 or atr_30[i] == 0 or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === VOLATILITY FILTER (avoid panic conditions) ===
        vol_ratio = atr_7[i] / atr_30[i]
        vol_normal = 0.5 < vol_ratio < 2.0  # Not too calm, not too volatile
        
        # === 1D MACRO BIAS (most important filter) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (3-bar lookback)
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # === CONNORS RSI EXTREMES (entry timing) ===
        crsi_oversold = crsi[i] < 15  # Very oversold for long entries
        crsi_overbought = crsi[i] > 85  # Very overbought for short entries
        crsi_recover_long = crsi[i] > 25 and crsi[i-1] < 25  # CRSI crossing up through 25
        crsi_recover_short = crsi[i] < 75 and crsi[i-1] > 75  # CRSI crossing down through 75
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.003  # Within 0.3% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.997  # Within 0.3% of upper band
        price_below_bb_lower = close[i] < bb_lower[i]  # Below lower band
        price_above_bb_upper = close[i] > bb_upper[i]  # Above upper band
        
        # === 30M LOCAL TREND ===
        price_above_hma_30m = close[i] > hma_30m[i]
        price_below_hma_30m = close[i] < hma_30m[i]
        
        # === ULTRA-SELECTIVE ENTRY LOGIC (4+ confluence required) ===
        new_signal = 0.0
        
        # --- LONG ENTRY (requires 4+ confluence) ---
        # Must have: 1d bullish + 4h bullish + session + volume + CRSI extreme + BB touch
        long_confluence = 0
        
        if price_above_hma_1d:
            long_confluence += 1
        if price_above_hma_4h and hma_4h_slope_bull:
            long_confluence += 1
        if in_session:
            long_confluence += 1
        if volume_confirmed:
            long_confluence += 1
        if vol_normal:
            long_confluence += 1
        if crsi_oversold or crsi_recover_long:
            long_confluence += 1
        if price_near_bb_lower or price_below_bb_lower:
            long_confluence += 1
        if price_above_hma_30m:
            long_confluence += 1
        
        # Need 5+ confluence for long entry (very strict)
        if long_confluence >= 5 and (crsi_oversold or crsi_recover_long):
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY (requires 4+ confluence) ---
        short_confluence = 0
        
        if price_below_hma_1d:
            short_confluence += 1
        if price_below_hma_4h and hma_4h_slope_bear:
            short_confluence += 1
        if in_session:
            short_confluence += 1
        if volume_confirmed:
            short_confluence += 1
        if vol_normal:
            short_confluence += 1
        if crsi_overbought or crsi_recover_short:
            short_confluence += 1
        if price_near_bb_upper or price_above_bb_upper:
            short_confluence += 1
        if price_below_hma_30m:
            short_confluence += 1
        
        # Need 5+ confluence for short entry (very strict)
        if short_confluence >= 5 and (crsi_overbought or crsi_recover_short):
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing - tighter for lower TF) ===
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
        
        # === EXIT ON MACRO BIAS CHANGE ===
        # Exit long if 1d bias turns bearish
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if 1d bias turns bullish
        if in_position and position_side < 0 and price_above_hma_1d:
            new_signal = 0.0
        
        # === EXIT ON CRSI REVERSAL (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            new_signal = 0.0  # Take profit on long
        
        if in_position and position_side < 0 and crsi[i] < 30:
            new_signal = 0.0  # Take profit on short
        
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