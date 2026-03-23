#!/usr/bin/env python3
"""
Experiment #075: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Session Filter

Hypothesis: 1h timeframe with 4h trend + 1d macro using Fisher Transform for reversal entries,
combined with session filter (8-20 UTC) and volume confirmation, will generate 30-80 trades/year
with Sharpe > 0.486. Fisher Transform excels at catching reversals in bear/range markets where
simple RSI fails.

Key innovations:
1) Ehlers Fisher Transform (period=9): long when crosses above -1.5, short when crosses below +1.5
2) 4h HMA(21) for intermediate trend direction
3) 1d HMA(21) for macro bias filter
4) Session filter: only trade 8-20 UTC (high liquidity, fewer false signals)
5) Volume filter: volume > 0.8x SMA(20) — not too strict to avoid 0 trades
6) Asymmetric sizing: 0.25 for trend-following, 0.20 for mean-reversion
7) ATR(14) stoploss at 2.5x with signal→0

Why this should work:
- 1h proven for entry timing (exp #065 failed due to too many trades, we add session filter)
- Fisher Transform catches reversals better than RSI in bear markets
- 4h/1d HMA prevents counter-trend trades
- Session filter reduces trade count by ~60% (only 12h of 24h)
- Volume filter relaxed (0.8x not 1.5x) to ensure trades on all symbols

Position size: 0.20-0.25 (discrete, smaller for 1h to reduce fee drag)
Stoploss: 2.5*ATR trailing
Target: 30-80 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_session_vol_4h1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    close_s = pd.Series(high + low).values / 2.0
    
    # Normalize price to range -1 to +1
    price_range = highest - lowest + 1e-10
    x = 0.67 * (close_s - lowest) / price_range - 0.67
    x = np.clip(x, -0.999, 0.999)  # Prevent division by zero in ln
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (previous Fisher value)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

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

def calculate_session_filter(open_time):
    """
    Filter for high-liquidity session (8-20 UTC).
    open_time is in milliseconds since epoch.
    """
    # Convert to datetime
    timestamps = pd.to_datetime(open_time, unit='ms', utc=True)
    hour = timestamps.dt.hour
    # Session: 8-20 UTC (inclusive of 8, exclusive of 20)
    session_active = (hour >= 8) & (hour < 20)
    return session_active.values

def calculate_volume_filter(volume, period=20, threshold=0.8):
    """Volume > threshold * SMA(volume)."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ok = volume > (threshold * vol_sma)
    return vol_ok

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
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    session_active = calculate_session_filter(open_time)
    vol_ok = calculate_volume_filter(volume, period=20, threshold=0.8)
    
    signals = np.zeros(n)
    POSITION_SIZE_TREND = 0.25
    POSITION_SIZE_MR = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER ===
        in_session = session_active[i]
        
        # === VOLUME FILTER ===
        volume_confirms = vol_ok[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ADAPTIVE ENTRY ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Fisher reversal + HTF bullish + session + volume ---
        if fisher_long and in_session and volume_confirms:
            # Trend-following long: 4h and 1d both bullish
            if price_above_hma_4h and price_above_hma_1d:
                new_signal = POSITION_SIZE_TREND
            # Mean-reversion long: 4h bullish OR 1d not strongly bearish
            elif price_above_hma_4h or (not price_below_hma_1d):
                if rsi_oversold:
                    new_signal = POSITION_SIZE_MR
        
        # --- SHORT ENTRY: Fisher reversal + HTF bearish + session + volume ---
        elif fisher_short and in_session and volume_confirms:
            # Trend-following short: 4h and 1d both bearish
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = -POSITION_SIZE_TREND
            # Mean-reversion short: 4h bearish OR 1d not strongly bullish
            elif price_below_hma_4h or (not price_above_hma_1d):
                if rsi_overbought:
                    new_signal = -POSITION_SIZE_MR
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not overbought and still in session
                if rsi_14[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not oversold
                if rsi_14[i] > 30.0:
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
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            # Exit long if both 4h and 1d turn bearish
            if price_below_hma_4h and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both 4h and 1d turn bullish
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