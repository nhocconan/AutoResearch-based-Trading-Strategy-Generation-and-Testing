#!/usr/bin/env python3
"""
Experiment #250: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume/Session

Hypothesis: After multiple 0-trade failures with Choppiness Index regime filters (#238, #240, #242),
return to simpler proven pattern: HTF trend direction + LTF pullback entries.
- 4h HMA(21) for intermediate trend direction
- 12h HMA(21) for macro bias confirmation
- 1h RSI(14) pullback entries (30-45 long, 55-70 short) — LOOSE thresholds for trade generation
- Volume > 0.8x 20-bar average (soft filter, not hard block)
- Session filter: 8-20 UTC only (major market hours)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 1h frequency)

TARGET: 40-80 trades/year on 1h, Sharpe > 0 on ALL symbols
CRITICAL: RSI thresholds MUST be loose (30-45, 55-70) to ensure >30 trades train, >3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
    # Calculate 4h HMA for intermediate trend (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 12h HMA for macro bias (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND DIRECTION (4h + 12h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bullish: price above both 4h and 12h HMA
        htf_bullish = price_above_hma_4h and price_above_hma_12h
        # Strong bearish: price below both 4h and 12h HMA
        htf_bearish = price_below_hma_4h and price_below_hma_12h
        # Neutral: mixed signals
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER (soft: > 0.8x average) ===
        volume_ok = volume[i] >= 0.8 * vol_avg_20[i]
        
        # === RSI PULLBACK ENTRY (LOOSE thresholds for trade generation) ===
        # Long: RSI 30-45 (oversold pullback in uptrend)
        rsi_long_entry = (rsi_14[i] >= 30.0) and (rsi_14[i] <= 45.0)
        # Short: RSI 55-70 (overbought pullback in downtrend)
        rsi_short_entry = (rsi_14[i] >= 55.0) and (rsi_14[i] <= 70.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bullish + RSI pullback + session + volume
        if htf_bullish and rsi_long_entry:
            if in_session and volume_ok:
                desired_signal = POSITION_SIZE
            elif in_session:  # session ok, volume weak - still enter with smaller size
                desired_signal = POSITION_SIZE * 0.6
            # Outside session: no entry
        
        # SHORT ENTRY: HTF bearish + RSI pullback + session + volume
        elif htf_bearish and rsi_short_entry:
            if in_session and volume_ok:
                desired_signal = -POSITION_SIZE
            elif in_session:
                desired_signal = -POSITION_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if HTF turns bearish
        if in_position and position_side > 0 and htf_bearish:
            desired_signal = 0.0
        
        # Exit short if HTF turns bullish
        if in_position and position_side < 0 and htf_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (mean reversion complete) ===
        # Exit long if RSI goes overbought (>70)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI goes oversold (<30)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish or neutral (not bearish)
                if htf_bullish or htf_neutral:
                    if rsi_14[i] <= 70.0:  # not overbought
                        desired_signal = POSITION_SIZE * 0.5
            elif position_side < 0:
                # Hold short if HTF still bearish or neutral (not bullish)
                if htf_bearish or htf_neutral:
                    if rsi_14[i] >= 30.0:  # not oversold
                        desired_signal = -POSITION_SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals