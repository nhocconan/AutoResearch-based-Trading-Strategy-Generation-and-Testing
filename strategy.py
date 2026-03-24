#!/usr/bin/env python3
"""
Experiment #1518: 30m Primary + 4h/1d HTF — Session-Filtered HMA Trend + RSI Pullback

Hypothesis: Lower TF (30m) can work IF we strictly control trade frequency via:
1. Dual HTF trend filter (4h HMA + 1d HMA) - both must agree
2. Session filter (8-20 UTC) - avoid Asian session whipsaw
3. Volume confirmation (>0.8x 20-bar avg) - ensure liquidity
4. RSI pullback within trend (not extreme) - quality entries
5. Smaller position size (0.20) - lower TF = more trades = need smaller size

Key learnings from 1100+ failed strategies:
- Complex regime filters (CHOP, CRSI) = 0 trades or negative Sharpe
- Volume-only filters = 0 trades (too restrictive)
- Session filter IS critical for lower TF (8-20 UTC avoids Asia chop)
- Dual HTF agreement (4h+1d) reduces false signals significantly
- 30m with proper filters can achieve 40-80 trades/year (sweet spot)

Design:
- 1d HMA(21) for macro trend (strictest filter)
- 4h HMA(21) for intermediate trend (confirmation)
- 30m RSI(14) for pullback entries (40-60 range = quality pullback)
- Session: only 8-20 UTC (European/US overlap = best liquidity)
- Volume: >0.8x 20-bar SMA (avoid low-liquidity traps)
- ATR(14) 2.5x trailing stop
- Position size: 0.20 (discrete: 0.0, ±0.20)

Timeframe: 30m (as required by experiment)
HTF: 4h + 1d (dual confirmation)
Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)
Target Sharpe: >0.618 (beat current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_session_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

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
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller size for 30m (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_30m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) - CRITICAL for lower TF ===
        hour = extract_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.8 * vol_sma[i]
        
        # === MACRO TREND (1d HMA) - strictest filter ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        fourh_bull = close[i] > hma_4h_aligned[i]
        fourh_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (30m HMA) - entry timing ===
        thirtym_bull = close[i] > hma_30m[i]
        thirtym_bear = close[i] < hma_30m[i]
        
        # === RSI PULLBACK - quality entries within trend ===
        # Long: RSI pulled back but not oversold (40-55 = healthy pullback)
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        # Short: RSI rallied but not overbought (45-60 = healthy rally)
        rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
        
        # === DESIRED SIGNAL - STRICT CONFLUENCE for 30m ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 4h bull + 30m bull + RSI pullback + session + volume
        # Option 1: Full confluence (all filters agree)
        if (daily_bull and fourh_bull and thirtym_bull and 
            rsi_pullback_long and in_session and volume_ok):
            desired_signal = BASE_SIZE
        # Option 2: HTF agreement + RSI pullback (slightly looser)
        elif (daily_bull and fourh_bull and rsi_pullback_long and in_session):
            desired_signal = BASE_SIZE * 0.8
        # Option 3: HTF agreement + 30m trend (fallback for trades)
        elif (daily_bull and fourh_bull and thirtym_bull and in_session and volume_ok):
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: 1d bear + 4h bear + 30m bear + RSI pullback + session + volume
        # Option 1: Full confluence (all filters agree)
        elif (daily_bear and fourh_bear and thirtym_bear and 
              rsi_pullback_short and in_session and volume_ok):
            desired_signal = -BASE_SIZE
        # Option 2: HTF agreement + RSI pullback (slightly looser)
        elif (daily_bear and fourh_bear and rsi_pullback_short and in_session):
            desired_signal = -BASE_SIZE * 0.8
        # Option 3: HTF agreement + 30m trend (fallback for trades)
        elif (daily_bear and fourh_bear and thirtym_bear and in_session and volume_ok):
            desired_signal = -BASE_SIZE * 0.6
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals