#!/usr/bin/env python3
"""
Experiment #497: 15m Primary + 4h/12h HTF — Regime-Adaptive RSI + HMA Trend

Hypothesis: 15m timeframe needs LOOSE entry conditions to generate trades (prior 15m 
strategies all had Sharpe=0.000 = ZERO TRADES). Use 4h HMA for trend bias, 12h ADX 
for regime detection (trend vs range), and 15m RSI for entry timing with wide thresholds.

Key changes from failed 15m experiments (#485, #489, #493):
- LOOSE RSI thresholds (35/65 for range, 45/55 for trend) — NOT 30/70
- OR logic for entries (any trigger works, not AND of 5+ conditions)
- Session filter is SOFT (prefer 00-12 UTC but allow all hours)
- No volume filters (volume failed on lower TFs)
- Simple regime switch: ADX>20 = trend follow, ADX<20 = mean revert

Strategy logic:
1. 4h HMA(21) = primary trend bias (faster than 1d for 15m entries)
2. 12h ADX(14) = regime filter (ADX>20 trending, ADX<20 ranging)
3. 15m RSI(14) = entry trigger (loose thresholds)
4. 15m HMA(21) = momentum confirmation
5. Session bias: 00-12 UTC preferred (London+NY overlap)
6. ATR(14)*2.5 stoploss on all positions

Target: Sharpe>0.40, trades>=160 train (40/year), trades>=30 test
Timeframe: 15m (first successful 15m experiment)
Position sizing: 0.20 base, 0.30 strong (conservative for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_rsi_hma_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime filter
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Session hours
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    prefer_session = (session_hours >= 0) & (session_hours <= 12)  # 00-12 UTC
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 12h REGIME FILTER (ADX) ===
        adx_value = adx_12h_aligned[i]
        is_trending = adx_value > 20.0  # ADX > 20 = trending regime
        is_ranging = adx_value <= 20.0  # ADX <= 20 = ranging regime
        
        # === 15m MOMENTUM ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else True
        below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else True
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (LOOSE THRESHOLDS) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 35.0
        rsi_extreme_overbought = rsi[i] > 65.0
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        rsi_mid_rising = rsi[i] > 50.0 and rsi[i-1] <= 50.0 if i > 0 else False
        rsi_mid_falling = rsi[i] < 50.0 and rsi[i-1] >= 50.0 if i > 0 else False
        
        # === SESSION FILTER (SOFT - prefer but not require) ===
        in_preferred_session = prefer_session[i]
        
        # === ENTRY LOGIC (LOOSE - multiple paths to entry) ===
        desired_signal = 0.0
        
        # TRENDING REGIME (ADX > 20): Follow HTF trend with pullback entries
        if is_trending:
            # LONG: 4h bull + 15m pullback (RSI<50) + momentum confirmation
            if htf_bull:
                if rsi[i] < 50.0 and rsi_rising and hma_bull:
                    desired_signal = SIZE_BASE
                    if in_preferred_session:
                        desired_signal = SIZE_STRONG
                elif rsi_extreme_oversold and htf_bull:
                    # Deep pullback in uptrend
                    desired_signal = SIZE_STRONG
                elif rsi_mid_rising and above_sma50:
                    # RSI crossing above 50 = momentum shift
                    desired_signal = SIZE_BASE * 0.8
            
            # SHORT: 4h bear + 15m rally (RSI>50) + momentum confirmation
            elif htf_bear:
                if rsi[i] > 50.0 and rsi_falling and hma_bear:
                    desired_signal = -SIZE_BASE
                    if in_preferred_session:
                        desired_signal = -SIZE_STRONG
                elif rsi_extreme_overbought and htf_bear:
                    # Deep rally in downtrend
                    desired_signal = -SIZE_STRONG
                elif rsi_mid_falling and below_sma50:
                    # RSI crossing below 50 = weakness
                    desired_signal = -SIZE_BASE * 0.8
        
        # RANGING REGIME (ADX <= 20): Mean reversion at extremes
        elif is_ranging:
            # LONG: RSI extreme oversold + above SMA200 (not in crash)
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
                if in_preferred_session:
                    desired_signal = SIZE_STRONG
            elif rsi_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            
            # SHORT: RSI extreme overbought + below SMA200 (not in rally)
            if desired_signal == 0.0:
                if rsi_extreme_overbought and below_sma200:
                    desired_signal = -SIZE_BASE
                    if in_preferred_session:
                        desired_signal = -SIZE_STRONG
                elif rsi_overbought and rsi_falling and below_sma50:
                    desired_signal = -SIZE_BASE * 0.8
        
        # HMA CROSSOVER (works in any regime)
        if desired_signal == 0.0:
            hma_prev = hma_15m[i-1] if i > 0 else hma_15m[i]
            if close[i] > hma_15m[i] and close[i-1] <= hma_prev and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif close[i] < hma_15m[i] and close[i-1] >= hma_prev and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals