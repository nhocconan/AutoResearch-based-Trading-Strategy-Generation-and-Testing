#!/usr/bin/env python3
"""
Experiment #1138: 30m Primary + 4h/1d HTF — Choppiness Regime + RSI Pullback

Hypothesis: After 828 failed experiments, the key insight is that lower TF (30m) 
strategies fail due to TOO MANY TRADES causing fee drag. This strategy uses:
1. 1d HMA(21) for macro trend direction (proven across all symbols)
2. 4h Choppiness Index for regime detection (CHOP > 55 = range, < 45 = trend)
3. 30m RSI(14) pullback entries within HTF trend
4. Volume filter (> 0.8x 20-period avg) to avoid low-liquidity traps
5. Session filter (8-20 UTC) for highest liquidity periods
6. ATR(14) 2.5x trailing stop for risk management

Why this should beat Sharpe=0.612:
- 30m entries within 4h/1d trend = HTF frequency with LTF precision
- Choppiness regime filter avoids whipsaw in ranging markets
- Session filter reduces noise from Asian session low-volume periods
- Conservative size (0.25) minimizes fee drag on lower TF
- Target: 40-80 trades/year (not 200+ like failed 30m strategies)

Timeframe: 30m (primary)
HTF: 4h, 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 (smaller for lower TF)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_rsi_hma_4h1d_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR first
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate choppiness
    range_val = hh - ll
    mask = (range_val > 1e-10) & (atr_sum > 1e-10)
    
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / range_val[mask]) / np.log10(period)
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Rolling average volume."""
    n = len(volume)
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').hour

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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h Choppiness Index for regime detection
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            continue
        if atr[i] <= 1e-10 or vol_avg[i] <= 1e-10:
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity periods
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        # Volume must be at least 0.8x average
        volume_ok = volume[i] >= 0.8 * vol_avg[i]
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        # CHOP < 45 = trending regime (follow trend)
        # CHOP > 55 = ranging regime (mean revert)
        trending_regime = chop_4h_aligned[i] < 45.0
        ranging_regime = chop_4h_aligned[i] > 55.0
        
        # === PULLBACK SIGNAL (30m RSI) ===
        # In trending regime: enter on pullback (RSI 40-60)
        # In ranging regime: enter at extremes (RSI < 35 or > 65)
        rsi_long_pullback = 35.0 < rsi_30m[i] < 50.0
        rsi_short_pullback = 50.0 < rsi_30m[i] < 65.0
        rsi_long_extreme = rsi_30m[i] < 35.0
        rsi_short_extreme = rsi_30m[i] > 65.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Trending regime: macro bull + RSI pullback
        # Ranging regime: macro bull + RSI extreme oversold
        if in_session and volume_ok:
            if trending_regime and macro_bull and rsi_long_pullback:
                desired_signal = BASE_SIZE
            elif ranging_regime and macro_bull and rsi_long_extreme:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Trending regime: macro bear + RSI pullback
        # Ranging regime: macro bear + RSI extreme overbought
        if in_session and volume_ok:
            if trending_regime and macro_bear and rsi_short_pullback:
                desired_signal = -BASE_SIZE
            elif ranging_regime and macro_bear and rsi_short_extreme:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals