#!/usr/bin/env python3
"""
Experiment #470: 1h Primary + 4h/1d HTF — Fisher Transform Reversal + HTF Trend

Hypothesis: 1h timeframe with Fisher Transform catches reversals better than RSI in 
bear/range markets (2025 test period). Fisher Transform normalizes price to Gaussian 
distribution, making extreme readings more reliable for mean reversion.

Key innovations vs failed 1h strategies (#459, #461, #465, #469):
1. FISHER TRANSFORM instead of RSI - better at catching bear market reversals
2. LOOSER entry thresholds - Fisher > -1.5 (long), Fisher < +1.5 (short)
3. 4h HMA for trend bias ONLY - not both 12h+1d (that was too restrictive)
4. Session filter 06-22 UTC (wider than 08-20) to ensure trades on all symbols
5. Single HTF (4h) instead of dual (12h+1d) - less filtering = more trades

Entry Logic:
- Long: 4h HMA bull + Fisher < -1.5 (oversold reversal) + session 06-22 UTC
- Short: 4h HMA bear + Fisher > +1.5 (overbought reversal) + session 06-22 UTC
- Exit: Fisher crosses back through 0 OR stoploss at 2.5x ATR

Why this should work:
- Fisher Transform reported Sharpe 0.8-1.5 through 2022 crash (research note)
- Works in bear markets where trend-following fails
- 4h trend filter prevents counter-trend trades in strong moves
- Session filter reduces trades to 40-80/year target

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_reversal_4h_trend_session_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching reversals than RSI in bear markets
    
    Steps:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize to -1 to +1 range using period high/low
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 using rolling high/low
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest > lowest and highest > 1e-10:
            # Normalize to -0.99 to +0.99 (avoid division by zero)
            normalized = 0.99 * (2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0)
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            if abs(normalized) < 0.99:
                fisher_raw[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    # Smooth Fisher with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_signal = pd.Series(fisher_raw).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    return fisher, fisher_signal

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_rsi(close, period=14):
    """Relative Strength Index - secondary filter"""
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength filter"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    # Convert to hours UTC: (ms / 1000 / 3600) % 24
    hours = ((open_time_ms / 1000.0 / 3600.0) % 24).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Session filter: 06-22 UTC (16 hours of high liquidity)
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (06-22 UTC) ===
        in_session = 6 <= hours[i] <= 22
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS) ===
        # Long: Fisher < -1.5 (oversold reversal zone)
        # Short: Fisher > +1.5 (overbought reversal zone)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher cross back through 0 for exit signal
        fisher_cross_zero_long = False
        fisher_cross_zero_short = False
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i-1] < 0 and fisher[i] >= 0:
                fisher_cross_zero_long = True
            if fisher[i-1] > 0 and fisher[i] <= 0:
                fisher_cross_zero_short = True
        
        # === RSI CONFIRMATION (SECONDARY FILTER) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === SMA FILTER (LONG-TERM TREND) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ADX FILTER (AVOID LOW VOLATILITY) ===
        adx_ok = adx[i] > 15.0  # Very loose - just avoid dead markets
        
        # === ENTRY LOGIC (LOOSE - 2-3 conditions max) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + Fisher oversold + session + (RSI or SMA200 confirmation)
        if htf_bull and fisher_oversold and in_session and adx_ok:
            # Need at least 1 more confirmation (RSI or SMA200)
            if rsi_oversold or above_sma200:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + Fisher overbought + session + (RSI or SMA200 confirmation)
        elif htf_bear and fisher_overbought and in_session and adx_ok:
            # Need at least 1 more confirmation (RSI or SMA200)
            if rsi_overbought or below_sma200:
                desired_signal = -SIZE_BASE
        
        # === EXIT CONDITIONS ===
        exit_signal = False
        
        if in_position and position_side > 0:
            # Long exit: Fisher crosses above 0 OR stoploss
            if fisher_cross_zero_long:
                exit_signal = True
        
        if in_position and position_side < 0:
            # Short exit: Fisher crosses below 0 OR stoploss
            if fisher_cross_zero_short:
                exit_signal = True
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered or exit_signal:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
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
        
        signals[i] = final_signal
    
    return signals