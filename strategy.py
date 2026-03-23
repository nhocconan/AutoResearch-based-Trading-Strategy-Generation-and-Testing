#!/usr/bin/env python3
"""
Experiment #668: 30m Primary + 4h/1d HTF — Trend-Follow with Volume + Session Filter

Hypothesis: 30m timeframe with 4h/1d HTF trend filter provides optimal balance for
lower TF trading. Key innovation: use HTF (4h HMA) for SIGNAL DIRECTION, 30m only
for ENTRY TIMING within HTF trend. This gives HTF trade frequency (~40-60/year)
with 30m execution precision.

Why this should work:
1. 4h HMA(21) for macro bias — prevents counter-trend trades (major failure cause)
2. 1d HMA(21) for secondary confirmation — extra filter for quality
3. 30m RSI(14) with MODERATE thresholds (35/65) — ensures trade generation
4. Volume filter (>0.8x 20-bar avg) — confirms institutional participation
5. Session filter (8-20 UTC) — avoids low-liquidity Asian session noise
6. ATR(14) trailing stop (2.5x) — protects capital on reversals
7. Position size 0.25 (lower for 30m TF to reduce fee drag)

Lessons from 442 failures:
- CRSI is TOO STRICT — generates 0 trades on BTC/ETH
- Choppiness Index alone doesn't work
- Need LOOSER RSI thresholds (35/65 not 20/80)
- ALL symbols must have positive Sharpe (no SOL-only)
- Lower TF MUST have HTF direction filter + strict entry

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period-1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    if n < period:
        return vol_ratio
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_avg + 1e-10)
    
    return vol_ratio

def get_hour_from_open_time(open_time):
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
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === HTF TREND BIAS (4h + 1d HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong HTF bias requires both 4h and 1d aligned
        htf_strong_bullish = htf_4h_bullish and htf_1d_bullish
        htf_strong_bearish = htf_4h_bearish and htf_1d_bearish
        
        # === VOLUME FILTER ===
        volume_confirmed = vol_ratio_30m[i] > 0.8
        
        # === RSI SIGNALS (MODERATE thresholds for trade generation) ===
        rsi_oversold = rsi_30m[i] < 40
        rsi_overbought = rsi_30m[i] > 60
        rsi_neutral = 40 <= rsi_30m[i] <= 60
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + RSI pullback + volume + session
        if htf_strong_bullish and rsi_oversold and volume_confirmed and in_session:
            desired_signal = SIZE_LONG
        # Also enter on RSI crossing up from oversold in bullish HTF
        elif htf_strong_bullish and rsi_30m[i] > 35 and rsi_30m[i] < 50 and volume_confirmed and in_session:
            desired_signal = SIZE_LONG
        
        # SHORT: HTF bearish + RSI rally + volume + session
        elif htf_strong_bearish and rsi_overbought and volume_confirmed and in_session:
            desired_signal = -SIZE_SHORT
        # Also enter on RSI crossing down from overbought in bearish HTF
        elif htf_strong_bearish and rsi_30m[i] < 65 and rsi_30m[i] > 50 and volume_confirmed and in_session:
            desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if HTF trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish AND RSI not extremely overbought
                if htf_4h_bullish and rsi_30m[i] < 70:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if 4h HMA still bearish AND RSI not extremely oversold
                if htf_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals