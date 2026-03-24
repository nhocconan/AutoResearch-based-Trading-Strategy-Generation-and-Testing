#!/usr/bin/env python3
"""
Experiment #510: 1h Primary + 4h/1d HTF — Confluence Pullback Strategy

Hypothesis: 1h timeframe with 4h trend + 1d bias + RSI pullback entries will generate
40-80 trades/year with positive Sharpe. Key insight from failed experiments:
- 15m/30m strategies got 0 trades (too strict filters)
- 6h/12h strategies got negative Sharpe (too slow for 2025 bear market)
- 1h is the sweet spot: responsive enough for 2025, slow enough to avoid fee drag

Strategy logic:
1. 1d HMA(21) = macro bias (only long if price > 1d HMA, only short if <)
2. 4h HMA(21) = intermediate trend direction
3. 1h RSI(14) = pullback entry timing (35-45 for long, 55-65 for short)
4. Session filter 08-20 UTC = reduce trades to 40-80/year
5. ATR(14)*2.5 stoploss on all positions
6. OR logic for entries (multiple trigger types, not all required)

Key changes from failed experiments:
- LOOSE RSI thresholds (35-45 not 30-35) to ensure trades generate
- Session filter to control trade count without killing opportunities
- 1h primary (not 15m/30m which got 0 trades, not 6h which was too slow)
- Simple HMA trend (not complex regime switching which failed)

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=20 test
Timeframe: 1h (as required by experiment #510)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    ema_1h = calculate_ema(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // (1000 * 3600)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === 1d HTF MACRO BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HTF TREND ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h HMA TREND ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI PULLBACK ZONES (LOOSE for trade generation) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 50.0
        rsi_oversold = rsi[i] < 40.0
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_cross_50_up = rsi[i] > 50.0 and rsi[i-1] <= 50.0 if i > 0 else False
        
        # Short: RSI rallied to 50-65 in downtrend
        rsi_pullback_short = 50.0 <= rsi[i] <= 65.0
        rsi_overbought = rsi[i] > 60.0
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        rsi_cross_50_down = rsi[i] < 50.0 and rsi[i-1] >= 50.0 if i > 0 else False
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 3.0  # Avoid extreme vol spikes
        
        # === ENTRY LOGIC (LOOSE - multiple triggers, OR logic) ===
        desired_signal = 0.0
        
        # TREND LONG: 1d bull + 4h bull + RSI pullback
        if htf_1d_bull and htf_4h_bull and vol_normal:
            if rsi_pullback_long and above_sma50:
                desired_signal = SIZE_BASE
            elif rsi_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE
            elif rsi_cross_50_up and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            elif hma_1h_bull and close[i] > ema_1h[i] and above_sma50:
                # HMA + EMA confluence
                desired_signal = SIZE_BASE * 0.8
        
        # TREND SHORT: 1d bear + 4h bear + RSI pullback
        elif htf_1d_bear and htf_4h_bear and vol_normal:
            if rsi_pullback_short and below_sma50:
                desired_signal = -SIZE_BASE
            elif rsi_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
            elif rsi_cross_50_down and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
            elif hma_1h_bear and close[i] < ema_1h[i] and below_sma50:
                # HMA + EMA confluence
                desired_signal = -SIZE_BASE * 0.8
        
        # MEAN REVERSION LONG: RSI extreme + SMA200 support (works in any regime)
        if desired_signal == 0.0 and vol_normal and in_session:
            if rsi[i] < 35.0 and above_sma200:
                desired_signal = SIZE_BASE * 0.8
            elif rsi[i] < 30.0 and above_sma50:
                # Very oversold
                desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: RSI extreme + SMA200 resistance (works in any regime)
        if desired_signal == 0.0 and vol_normal and in_session:
            if rsi[i] > 65.0 and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
            elif rsi[i] > 70.0 and below_sma50:
                # Very overbought
                desired_signal = -SIZE_BASE
        
        # STRONG TREND: All HTF aligned + momentum
        if desired_signal == 0.0 and vol_normal and in_session:
            if htf_1d_bull and htf_4h_bull and hma_1h_bull and rsi[i] > 50.0 and rsi[i] < 65.0:
                desired_signal = SIZE_STRONG
            elif htf_1d_bear and htf_4h_bear and hma_1h_bear and rsi[i] < 50.0 and rsi[i] > 35.0:
                desired_signal = -SIZE_STRONG
        
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
        
        # === APPLY SESSION FILTER TO FINAL SIGNAL ===
        # Only allow entries during session, but let stops work anytime
        if desired_signal != 0.0 and not in_session and not in_position:
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