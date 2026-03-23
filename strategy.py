#!/usr/bin/env python3
"""
Experiment #748: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 500+ failed strategies, clear patterns emerge for lower TF success:
1. 30m MUST use HTF for direction (4h/1d HMA), 30m only for entry timing
2. Complex regime (Chop/CRSI) causes 0 trades — use simpler RSI pullback instead
3. Session filter (8-20 UTC) + Volume filter reduces false breakouts significantly
4. Target: 30-80 trades/year (strict entry confluence: HTF trend + RSI + Volume + Session)
5. Position size: 0.25 (conservative for lower TF), discrete levels only

Strategy design:
1. 1d HMA(21) for primary trend bias (proven in best strategies like #739, #743)
2. 4h HMA(16) for intermediate trend confirmation
3. 30m RSI(7) pullback entries (RSI 35-45 for long, 55-65 for short) — NOT extremes
4. Volume filter: current volume > 1.2x 20-bar average (confirms breakout)
5. Session filter: only trade 8-20 UTC (avoid Asia low-liquidity whipsaws)
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signals: 0.0, ±0.25 (minimize fee churn)

Key differences from failed #738 (30m RSI pullback):
- Added 4h HMA confirmation (not just 1d)
- Added volume filter (was missing in #738)
- Added session filter (was missing in #738)
- Looser RSI ranges (35-45 not 25-35) to ensure trade frequency
- Clear hold logic to maintain positions through trends

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 30m (target 30-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_vol_session_v2"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
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
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=7)  # Faster RSI for pullback detection
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 30m TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20  # London/NY overlap, avoid Asia low liquidity
        
        # Volume filter: current volume > 1.2x 20-bar average
        volume_ok = volume[i] > 1.2 * vol_sma_30m[i]
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK DETECTION (30m) ===
        # Long: RSI pulled back to 35-45 zone in uptrend
        rsi_pullback_long = 35 <= rsi_30m[i] <= 48
        # Short: RSI rallied to 52-65 zone in downtrend
        rsi_pullback_short = 52 <= rsi_30m[i] <= 65
        
        # === MOMENTUM CONFIRMATION ===
        # Price above both HTF HMAs for long confirmation
        strong_bullish = trend_1d_bullish and trend_4h_bullish
        # Price below both HTF HMAs for short confirmation
        strong_bearish = trend_1d_bearish and trend_4h_bearish
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (ALL must be true) ===
        long_signal = (
            strong_bullish and  # Both 1d and 4h trend bullish
            rsi_pullback_long and  # RSI pullback zone
            in_session and  # Trading session filter
            volume_ok  # Volume confirmation
        )
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (ALL must be true) ===
        short_signal = (
            strong_bearish and  # Both 1d and 4h trend bearish
            rsi_pullback_short and  # RSI pullback zone
            in_session and  # Trading session filter
            volume_ok  # Volume confirmation
        )
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both signals (rare), go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === HOLD LOGIC — Maintain position if HTF trends intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d and 4h HMA still bullish
                if trend_1d_bullish and trend_4h_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d and 4h HMA still bearish
                if trend_1d_bearish and trend_4h_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS — HTF trend reversal ===
        if in_position and position_side > 0:
            # Exit long if 1d or 4h trend reverses
            if trend_1d_bearish or trend_4h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d or 4h trend reverses
            if trend_1d_bullish or trend_4h_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
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