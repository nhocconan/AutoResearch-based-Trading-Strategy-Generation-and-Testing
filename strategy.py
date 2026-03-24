#!/usr/bin/env python3
"""
Experiment #577: 15m Primary + 4h/12h HTF — RSI Pullback Trend Following

Hypothesis: 15m timeframe with 4h HMA trend bias + 12h regime filter + 15m RSI pullback entries
will capture intraday trends while avoiding whipsaw. Key insight from failures:
- 15m needs HTF direction filter (4h HMA) to avoid counter-trend trades
- 12h Choppiness tells us if we're in trend or range regime
- 15m RSI pullback (not extreme) gives entry timing without being too restrictive
- Session filter (UTC 08-20) avoids low-volume Asian session whipsaw

Why this should work:
1. 4h HMA(21) = slow trend filter (only trade in HTF trend direction)
2. 12h CHOP(14) = regime detection (trend follow when CHOP<45, mean-revert when CHOP>55)
3. 15m RSI(7) pullback to 40-60 zone = entry timing (not extreme RSI)
4. ATR(14)*2.5 stoploss = risk management
5. Session filter = avoid low-volume hours

Target: 50-100 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_hma4h_chop12h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    
    # Calculate and align 12h Choppiness for regime filter
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 15m HMA for additional trend confirmation
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC hours from open_time) ===
        # Convert open_time (milliseconds) to hour
        hour_utc = (open_time[i] // 3600000) % 24
        # Prefer London/NY overlap (08:00-20:00 UTC), avoid Asian session (00:00-06:00)
        is_good_session = 7 <= hour_utc <= 22
        
        # === 4H TREND BIAS ===
        hma_4h = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h
        
        trend_bull = close[i] > hma_4h and hma_4h > hma_4h_prev
        trend_bear = close[i] < hma_4h and hma_4h < hma_4h_prev
        trend_neutral = not trend_bull and not trend_bear
        
        # === 12H REGIME (Choppiness) ===
        chop = chop_12h_aligned[i]
        is_trend_regime = chop < 50.0  # Trending market
        is_range_regime = chop > 55.0  # Range-bound market
        is_transition = not is_trend_regime and not is_range_regime
        
        # === 15M RSI PULLBACK ===
        rsi_val = rsi[i]
        rsi_prev = rsi[i-1] if i > 0 else rsi_val
        
        # Pullback zones (not extremes - we want frequent entries)
        rsi_bull_pullback = 35.0 <= rsi_val <= 55.0 and rsi_val > rsi_prev
        rsi_bear_pullback = 45.0 <= rsi_val <= 65.0 and rsi_val < rsi_prev
        
        # Extreme reversals (for range regime)
        rsi_oversold = rsi_val < 30.0
        rsi_overbought = rsi_val > 70.0
        
        # === 15M HMA CONFIRMATION ===
        hma_15m_val = hma_15m[i]
        hma_15m_prev = hma_15m[i-5] if i >= 5 else hma_15m_val
        
        hma_15m_bull = close[i] > hma_15m_val and hma_15m_val > hma_15m_prev
        hma_15m_bear = close[i] < hma_15m_val and hma_15m_val < hma_15m_prev
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow 4h trend with 15m RSI pullback entry
        if is_trend_regime and is_good_session:
            if trend_bull and (rsi_bull_pullback or rsi_oversold) and hma_15m_bull:
                desired_signal = SIZE_STRONG
            elif trend_bear and (rsi_bear_pullback or rsi_overbought) and hma_15m_bear:
                desired_signal = -SIZE_STRONG
            # Simpler: just trend + RSI in favorable zone
            elif trend_bull and rsi_val < 50.0:
                desired_signal = SIZE_BASE
            elif trend_bear and rsi_val > 50.0:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif is_range_regime and is_good_session:
            if rsi_oversold and close[i] > hma_4h:
                desired_signal = SIZE_BASE
            elif rsi_overbought and close[i] < hma_4h:
                desired_signal = -SIZE_BASE
            # RSI recovery
            elif rsi_val < 35.0 and rsi_val > rsi_prev:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_val > 65.0 and rsi_val < rsi_prev:
                desired_signal = -SIZE_BASE * 0.8
        
        # TRANSITION REGIME: Reduced size, wait for clearer signals
        elif is_transition and is_good_session:
            if trend_bull and rsi_oversold:
                desired_signal = SIZE_BASE * 0.6
            elif trend_bear and rsi_overbought:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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