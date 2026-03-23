#!/usr/bin/env python3
"""
Experiment #228: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume Session

Hypothesis: After 30m failures with 0 trades (#218, #225), simplify entry logic.
Use 4h HMA for TREND DIRECTION (not entry trigger), 30m RSI for ENTRY TIMING only.
Key insight: Don't require all filters to align perfectly — use OR logic for entries
within HTF trend direction to ensure adequate trade frequency.

Lessons from failures:
- #218 (30m hma rsi vol session): 0 trades — filters too strict/ mutually exclusive
- #225 (1h crsi chop hma): 0 trades — CRSI+Choppiness combination too restrictive
- Complex regime switching fails on lower TF (#222, #223, #227 all negative Sharpe)

New approach:
1. 4h HMA(21) = primary trend bias (long only when price > 4h HMA)
2. 1d HMA(21) = macro filter (increase size when aligned with 1d)
3. 30m RSI(14) pullback = entry trigger (RSI<45 for long, RSI>55 for short)
4. Volume > 0.7x 20-bar avg (looser than 0.8x to get more trades)
5. Session 8-20 UTC (liquid hours only)
6. ATR(14) 2.5x trailing stoploss

TARGET: 40-80 trades/year, Sharpe > 0.4 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.20, ±0.30 (discrete, max 0.30 for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_session_4h1d_atr_v2"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    n = len(close)
    close_s = pd.Series(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

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
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
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
    
    # Calculate 30m indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # Calculate 4h HMA for trend direction (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20  # Liquid hours only
        
        # Volume filter (looser: 0.7x instead of 0.8x to get more trades)
        volume_ok = volume[i] >= 0.7 * vol_sma_20[i]
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK SIGNALS (30m) ===
        # Looser thresholds to ensure trades: 35-65 instead of 40-60
        rsi_oversold = rsi_14[i] < 45.0  # Long entry zone
        rsi_overbought = rsi_14[i] > 55.0  # Short entry zone
        rsi_neutral = 40.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC (OR logic to ensure trades) ===
        new_signal = 0.0
        
        # LONG: 4h trend up + RSI pullback OR 4h trend up + RSI neutral + volume
        if price_above_hma_4h:
            if rsi_oversold and in_session:
                # Strong long signal (RSI oversold in uptrend)
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_FULL  # Aligned with 1d
                else:
                    new_signal = POSITION_SIZE_HALF  # Against 1d macro
            elif rsi_neutral and volume_ok and in_session:
                # Weaker long signal (neutral RSI but volume confirmation)
                new_signal = POSITION_SIZE_HALF
        
        # SHORT: 4h trend down + RSI pullback OR 4h trend down + RSI neutral + volume
        elif price_below_hma_4h:
            if rsi_overbought and in_session:
                # Strong short signal (RSI overbought in downtrend)
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_FULL  # Aligned with 1d
                else:
                    new_signal = -POSITION_SIZE_HALF  # Against 1d macro
            elif rsi_neutral and volume_ok and in_session:
                # Weaker short signal (neutral RSI but volume confirmation)
                new_signal = POSITION_SIZE_HALF * -1
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (don't exit on every bar)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still bullish
                if price_above_hma_4h and rsi_14[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still bearish
                if price_below_hma_4h and rsi_14[i] > 30.0:
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
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
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