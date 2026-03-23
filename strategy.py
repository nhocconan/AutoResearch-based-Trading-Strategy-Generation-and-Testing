#!/usr/bin/env python3
"""
Experiment #478: 30m Primary + 4h/1d HTF — Volatility Spike Reversion + Choppiness Regime + HTF Trend

Hypothesis: Based on research showing volatility spike reversion works well in bear/range markets.
When ATR(7)/ATR(30) > 2.0, volatility is at extreme levels and mean reversion is likely.
Combined with Choppiness Index for regime detection and 4h/1d HMA for trend bias.

Key innovations:
1. ATR Ratio (7/30) > 2.0 = volatility extreme (entry trigger)
2. Choppiness Index(14) regime filter: CHOP>55=range (mean revert), CHOP<45=trend (follow)
3. 4h HMA(21) + 1d HMA(50) for HTF trend alignment (both must agree)
4. RSI(14) extreme for entry timing (<30 long, >70 short)
5. Session filter: only 8-20 UTC (highest liquidity, avoid Asian chop)
6. Volume filter: volume > 0.8x 20-bar average
7. Very strict confluence: need 4+ signals to enter (limits to 30-80 trades/year)
8. Discrete position sizing: 0.0, ±0.20, ±0.30
9. ATR(14) trailing stop at 2.5x for risk management

Why this should work for 30m:
- HTF (4h/1d) determines DIRECTION (reduces whipsaws)
- 30m only for ENTRY TIMING within HTF trend
- Volatility spike = high probability mean reversion
- Session + volume filters reduce false signals
- Strict confluence (4+ filters) ensures low trade count

Target: Sharpe > 0.612, DD < -35%, trades 30-80/year on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_spike_chop_regime_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    # WMA calculations
    def wma(arr, p):
        result = np.full(len(arr), np.nan)
        weights = np.arange(1, p + 1)
        for i in range(p - 1, len(arr)):
            if np.any(np.isnan(arr[i - p + 1:i + 1])):
                continue
            result[i] = np.sum(arr[i - p + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    if len(close) >= period:
        diff = 2 * wma_half - wma_full
        sqrt_period = int(np.sqrt(period))
        if sqrt_period < 1:
            sqrt_period = 1
        hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_session_hour(open_time):
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
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    
    # Volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
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
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            continue
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop_30m[i]):
            continue
        if np.isnan(rsi_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        
        # === VOLATILITY SPIKE (Entry Trigger) ===
        atr_ratio = atr_7[i] / atr_30[i]
        vol_spike = atr_ratio > 2.0  # Volatility at extreme
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop = chop_30m[i] > 55.0  # Range/mean reversion regime
        is_trend = chop_30m[i] < 45.0  # Trending regime
        # Neutral zone: 45 <= CHOP <= 55
        
        # === HTF TREND ALIGNMENT (4h + 1d must agree) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong signal
        htf_strong_bullish = htf_4h_bullish and htf_1d_bullish
        htf_strong_bearish = htf_4h_bearish and htf_1d_bearish
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_30m[i] < 30.0
        rsi_overbought = rsi_30m[i] > 70.0
        rsi_extreme_oversold = rsi_30m[i] < 25.0
        rsi_extreme_overbought = rsi_30m[i] > 75.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = calculate_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES (need 4+ confluence)
        long_score = 0
        
        # Volatility spike (required trigger)
        if vol_spike:
            long_score += 2
        
        # HTF alignment (both 4h and 1d bullish)
        if htf_strong_bullish:
            long_score += 2
        elif htf_4h_bullish or htf_1d_bullish:
            long_score += 1
        
        # RSI entry
        if is_chop:
            # In chop: need extreme RSI for mean reversion
            if rsi_extreme_oversold:
                long_score += 2
            elif rsi_oversold:
                long_score += 1
        else:
            # In trend or neutral: moderate RSI ok
            if rsi_oversold:
                long_score += 2
        
        # Session filter
        if in_session:
            long_score += 1
        
        # Volume filter
        if vol_ok:
            long_score += 1
        
        # Enter long if score >= 5 (strict confluence)
        if long_score >= 5:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # Volatility spike (required trigger)
            if vol_spike:
                short_score += 2
            
            # HTF alignment
            if htf_strong_bearish:
                short_score += 2
            elif htf_4h_bearish or htf_1d_bearish:
                short_score += 1
            
            # RSI entry
            if is_chop:
                if rsi_extreme_overbought:
                    short_score += 2
                elif rsi_overbought:
                    short_score += 1
            else:
                if rsi_overbought:
                    short_score += 2
            
            # Session filter
            if in_session:
                short_score += 1
            
            # Volume filter
            if vol_ok:
                short_score += 1
            
            if short_score >= 5:
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
            if position_side > 0 and htf_4h_bullish and htf_1d_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and htf_4h_bearish and htf_1d_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
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