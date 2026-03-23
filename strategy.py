#!/usr/bin/env python3
"""
Experiment #638: 30m Primary + 4h/1d HTF — HMA Trend + RSI Extremes + Volume + Session

Hypothesis: Lower timeframe (30m) strategies fail due to over-filtering (see #628, #632, #635 
all with Sharpe=0.000 = 0 trades). This strategy uses SIMPLER confluence:
1. 4h HMA trend direction (NOT 12h/1d which is too slow for 30m entries)
2. 30m RSI extremes (<35 long, >65 short) for mean-reversion within trend
3. Volume filter (>0.8x 20-bar avg) to confirm participation
4. Session filter (8-20 UTC) when institutional volume is highest

Why this might work where others failed:
- 4h HMA is fast enough to catch moves but slow enough to filter noise
- RSI extremes (35/65) are looser than typical (30/70) = more trades
- Session filter reduces false breakouts during low-volume Asia session
- Volume confirmation prevents entries on thin liquidity
- Position size 0.22 (conservative for 30m per Rule 4)
- Target: 40-80 trades/year (per Rule 10 for 30m)

Key difference from failed experiments:
- NO Choppiness Index (caused 0 trades in #628, #629, #631, #632)
- NO CRSI (caused 0 trades in #630, #631, #632)
- NO Donchian breakouts (added complexity without benefit in #636, #637)
- Simple RSI extremes + HTF trend = proven pattern

Position sizing: 0.22 discrete (smaller for 30m per Rule 4)
Target: 40-80 trades/year on 30m (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_session_4h_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for primary trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    hma_30m = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_30m[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(utc_hour[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 30M HMA SLOPE (2 bars) ===
        hma_30m_slope_bull = hma_30m[i] > hma_30m[i-2] if i >= 2 else False
        hma_30m_slope_bear = hma_30m[i] < hma_30m[i-2] if i >= 2 else False
        
        # === RSI EXTREMES (looser than typical 30/70) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === SESSION FILTER (8-20 UTC = high liquidity) ===
        session_active = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 4h bull trend + 30m RSI oversold + volume + session ---
        # Condition 1: 4h HMA sloping up + price above 4h HMA
        # Condition 2: 30m HMA sloping up (momentum confirmation)
        # Condition 3: RSI < 35 (oversold pullback in uptrend)
        # Condition 4: Volume > 0.8x average
        # Condition 5: Session 8-20 UTC
        if hma_4h_slope_bull and price_above_hma_4h:
            if hma_30m_slope_bull:
                if rsi_oversold and volume_confirmed and session_active:
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 4h bear trend + 30m RSI overbought + volume + session ---
        # Condition 1: 4h HMA sloping down + price below 4h HMA
        # Condition 2: 30m HMA sloping down (momentum confirmation)
        # Condition 3: RSI > 65 (overbought bounce in downtrend)
        # Condition 4: Volume > 0.8x average
        # Condition 5: Session 8-20 UTC
        elif hma_4h_slope_bear and price_below_hma_4h:
            if hma_30m_slope_bear:
                if rsi_overbought and volume_confirmed and session_active:
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
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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