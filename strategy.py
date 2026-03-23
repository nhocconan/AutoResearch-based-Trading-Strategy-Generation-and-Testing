#!/usr/bin/env python3
"""
Experiment #188: 30m Primary + 4h/1d HTF — Regime-Adaptive with Session Filter

Hypothesis: Lower timeframe (30m) strategies fail due to either (1) too many trades
causing fee drag, or (2) too strict filters causing 0 trades. This strategy uses:
- 4h HMA for signal DIRECTION (trend bias)
- 1d HMA for macro confirmation (avoid counter-trend)
- 30m Connors RSI for ENTRY TIMING (pullback entries within HTF trend)
- Choppiness Index for regime detection (range vs trend mode)
- Volume filter (>0.8x 20-period avg) to confirm moves
- Session filter (8-20 UTC) for high liquidity hours only

Key innovations:
1. HTF (4h/1d) determines direction, LTF (30m) only times entry
2. Regime-adaptive: mean reversion in chop, trend follow otherwise
3. Session filter reduces noise during low-liquidity hours
4. Conservative position sizing (0.20-0.25) for lower TF fee management
5. Looser CRSI thresholds (12/88 instead of 10/90) to ensure trades

TARGET: 40-70 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_session_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 12, Short when CRSI > 88
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0).values
    
    # RSI of Streak (consecutive up/down bars)
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
            streak_rsi[i] = 100.0 * min(streak_abs[i], streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(streak_abs[i], streak_period) / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            rank = np.sum(returns[:-1] < returns[-1])
            percent_rank[i] = 100.0 * rank / (len(returns) - 1)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Extract session hours
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER (>0.8x average) ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === HTF MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours with volume confirmation
        if in_session and volume_ok:
            if is_range:
                # MEAN REVERSION MODE (Connors RSI pullbacks)
                # Long: CRSI < 12 + 4h HMA bullish + 1d HMA confirms
                if crsi[i] < 12.0 and price_above_hma_4h:
                    if price_above_hma_1d:
                        new_signal = POSITION_SIZE_FULL
                    else:
                        new_signal = POSITION_SIZE_HALF
                
                # Short: CRSI > 88 + 4h HMA bearish + 1d HMA confirms
                elif crsi[i] > 88.0 and price_below_hma_4h:
                    if price_below_hma_1d:
                        new_signal = -POSITION_SIZE_FULL
                    else:
                        new_signal = -POSITION_SIZE_HALF
            
            elif is_trend:
                # TREND FOLLOWING MODE (pullback to HMA)
                # Long: Price pulls back to 4h HMA + CRSI recovering from oversold
                if price_above_hma_4h and crsi[i] < 40.0 and crsi[i] > 20.0:
                    if price_above_hma_1d:
                        new_signal = POSITION_SIZE_FULL
                    else:
                        new_signal = POSITION_SIZE_HALF
                
                # Short: Price pulls back to 4h HMA + CRSI recovering from overbought
                elif price_below_hma_4h and crsi[i] > 60.0 and crsi[i] < 80.0:
                    if price_below_hma_1d:
                        new_signal = -POSITION_SIZE_FULL
                    else:
                        new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and HTF trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h HMA
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h HMA
                if price_below_hma_4h:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA (trend changed)
        if in_position and position_side < 0 and price_above_hma_4h:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals