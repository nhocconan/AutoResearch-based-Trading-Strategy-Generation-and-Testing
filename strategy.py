#!/usr/bin/env python3
"""
Experiment #516: 30m Primary + 4h/1d HTF — Choppiness Regime + RSI Pullback

Hypothesis: 30m timeframe with 4h/1d HTF provides optimal balance for
capturing intraday swings while respecting higher-timeframe trend.
Choppiness Index regime filter switches between mean-reversion (chop>61.8)
and trend-following (chop<38.2). RSI pullback entries in direction of HTF trend.
Session filter (08-20 UTC) reduces trades during low-volume Asian session.

Strategy logic:
1. 1d HMA(21) = daily trend bias (primary HTF filter)
2. 4h HMA(21) = intermediate trend confirmation
3. 30m Choppiness(14) = regime detection (range vs trend)
4. 30m RSI(7) = entry timing (pullback in trend, extremes in range)
5. 30m Session filter (08-20 UTC) = avoid Asian session whipsaws
6. ATR(14)*2.5 stoploss on all positions
7. Regime-adaptive: mean revert in chop, trend follow otherwise

Key improvements from failed experiments:
- Simpler RSI(7) instead of Connors RSI (faster response on 30m)
- Session filter to reduce trade count (target 40-80/year)
- Looser RSI thresholds (25/75 instead of 10/90) for more trades
- Both 4h AND 1d HTF must agree for trend entries (stronger confluence)

Target: Sharpe>0.50, trades>=120 train (30/year), trades>=15 test
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_rsi_session_4h1d_v1"
timeframe = "30m"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds, convert to hour
    hours = (prices["open_time"].values // (1000 * 60 * 60)) % 24
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 30m
    chop = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
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
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Only trade during high-volume hours (London/NY overlap)
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === 1d HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF CONFLUENCE (both must agree for strong signal) ===
        htf_strong_bull = htf_1d_bull and htf_4h_bull
        htf_strong_bear = htf_1d_bear and htf_4h_bear
        htf_agree = htf_strong_bull or htf_strong_bear
        
        # === 30m HMA TREND ===
        hma_30m = calculate_hma(close[:i+1], 21)[-1] if i >= 20 else np.nan
        hma_bull = close[i] > hma_30m if not np.isnan(hma_30m) else False
        hma_bear = close[i] < hma_30m if not np.isnan(hma_30m) else False
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0  # Range-bound market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi[i] < 30.0  # Loose threshold for more trades
        rsi_overbought = rsi[i] > 70.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        rsi_neutral = (rsi[i] >= 40.0) and (rsi[i] <= 60.0)
        
        # RSI rising/falling
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # RSI pullback from extreme (entering from oversold/overbought)
        rsi_pullback_long = rsi_extreme_oversold and rsi_rising
        rsi_pullback_short = rsi_extreme_overbought and rsi_falling
        
        # === VOLATILITY FILTER ===
        if i >= 100:
            atr_avg = np.nanmean(atr[max(0,i-100):i])
            atr_ratio = atr[i] / atr_avg if atr_avg > 1e-10 else 1.0
        else:
            atr_ratio = 1.0
        vol_normal = atr_ratio < 3.0  # Avoid extreme vol spikes
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # TREND REGIME: Follow HTF direction with RSI pullback
        if chop_trend and vol_normal and htf_agree:
            if htf_strong_bull and rsi_pullback_long and above_sma50:
                desired_signal = SIZE_STRONG
            elif htf_strong_bear and rsi_pullback_short and below_sma50:
                desired_signal = -SIZE_STRONG
            # RSI crossing up from neutral in uptrend
            elif htf_strong_bull and rsi[i] > 45.0 and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE
            # RSI crossing down from neutral in downtrend
            elif htf_strong_bear and rsi[i] < 55.0 and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion with RSI extremes
        if chop_range and vol_normal:
            if rsi_extreme_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_extreme_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            # RSI recovery from extreme
            elif rsi_oversold and rsi_rising and above_sma50:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_overbought and rsi_falling and below_sma50:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME (chop 45-55): Use HTF bias with RSI
        if not chop_range and not chop_trend and vol_normal:
            if htf_strong_bull and rsi_oversold and rsi_rising:
                desired_signal = SIZE_BASE
            elif htf_strong_bear and rsi_overbought and rsi_falling:
                desired_signal = -SIZE_BASE
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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