#!/usr/bin/env python3
"""
Experiment #488: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 400+ failed experiments, complex regime-switching (CRSI, Choppiness) leads to 0 trades.
Simple MTF trend-following with relaxed thresholds (#486: Sharpe=0.440) works better. For 30m TF,
we need STRICT filters to avoid fee drag (>100 trades/year kills profit).

Key innovations:
1. 1d HMA(21) for MAJOR trend bias (proven in #486)
2. 4h HMA(21) for INTERMEDIATE trend confirmation (new layer between 30m and 1d)
3. 30m RSI(14) for ENTRY TIMING only (pullback in HTF trend direction)
4. Session filter: 8-20 UTC (high liquidity, avoid Asian session noise)
5. Volume filter: > 0.8x 20-bar avg (confirms institutional participation)
6. ATR(14) trailing stop at 2.5x (protects against 2022-style crashes)
7. Discrete sizing: 0.25 long/short (smaller for 30m to reduce fee impact)
8. HOLD logic: maintain position while HTF trend intact (reduces churn)

Why 30m with strict filters: More entry precision than 4h/12h, but session+volume filters
keep trades at 30-80/year target. 4h intermediate trend reduces whipsaws vs 30m-only.

Target: Sharpe > 0.612, DD < -35%, trades 30-80/year, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    if half < 1 or sqrt_period < 1:
        return hma
    
    def wma(series, w_period):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        for i in range(w_period - 1, len(series)):
            if np.any(np.isnan(series[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(series[i - w_period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
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

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

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
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.8 * vol_avg_20[i]
        
        # === HTF MAJOR TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        mid_bullish = close[i] > hma_4h_aligned[i]
        mid_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK SIGNALS (entry timing) ===
        # Long: RSI pulled back but not oversold (40-55 range in uptrend)
        rsi_pullback_long = 35.0 <= rsi_30m[i] <= 55.0
        # Short: RSI rallied but not overbought (45-65 range in downtrend)
        rsi_pullback_short = 45.0 <= rsi_30m[i] <= 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - All conditions must align
        if in_session and volume_ok:
            long_score = 0
            
            # HTF bullish bias (required)
            if htf_bullish:
                long_score += 3
            
            # 4h intermediate bullish (required for 30m entries)
            if mid_bullish:
                long_score += 2
            
            # RSI pullback (entry timing)
            if rsi_pullback_long:
                long_score += 2
            
            # Enter long if score >= 6 (all major conditions)
            if long_score >= 6:
                desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0 and in_session and volume_ok:
            short_score = 0
            
            # HTF bearish bias (required)
            if htf_bearish:
                short_score += 3
            
            # 4h intermediate bearish
            if mid_bearish:
                short_score += 2
            
            # RSI pullback
            if rsi_pullback_short:
                short_score += 2
            
            if short_score >= 6:
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
                # Hold long if 1d still bullish OR 4h still bullish
                if htf_bullish or mid_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if 1d still bearish OR 4h still bearish
                if htf_bearish or mid_bearish:
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
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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