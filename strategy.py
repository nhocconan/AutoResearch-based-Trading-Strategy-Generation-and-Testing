#!/usr/bin/env python3
"""
Experiment #665: 15m Primary + 4h/1d HTF — Daily CPR Breakout + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with Daily CPR (Central Pivot Range) from 1d HTF provides key support/resistance 
levels. Combined with 4h HMA trend filter and RSI(7) pullback entries, this captures intraday breakouts 
in direction of higher timeframe trend. Session filter (00-12 UTC) reduces noise during low-volume periods.

Key innovations:
1. Daily CPR levels from 1d HTF - Pivot, BC (Bottom Central), TC (Top Central)
2. CPR Width filter - narrow CPR (<30th percentile) signals impending breakout
3. 4h HMA(21) trend bias - only long above, only short below
4. RSI(7) pullback entry - enter on oversold bounce in uptrend, overbought fade in downtrend
5. Session filter - prefer 00-12 UTC (London/NY overlap for crypto liquidity)
6. ATR(14) trailing stop - 2.5x for risk management

Entry conditions (balanced for trade frequency):
- LONG: price > 1d TC + narrow CPR + 4h HMA bull + RSI(7) < 40 (pullback) + session filter
- SHORT: price < 1d BC + narrow CPR + 4h HMA bear + RSI(7) > 60 (pullback) + session filter
- Breakout: price breaks TC/BC with expanding CPR + 4h trend confirmation

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match original length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
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

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_cpr_levels(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) levels from 1d data
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    CPR Width = TC - BC
    """
    n = len(df_1d)
    pivot = np.full(n, np.nan)
    bc = np.full(n, np.nan)
    tc = np.full(n, np.nan)
    cpr_width = np.full(n, np.nan)
    
    for i in range(n):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        
        pivot[i] = (h + l + c) / 3.0
        bc[i] = (h + l) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
        cpr_width[i] = tc[i] - bc[i]
    
    return pivot, bc, tc, cpr_width

def calculate_cpr_width_percentile(cpr_width, lookback=30):
    """Calculate rolling percentile of CPR width to detect narrow/wide CPR"""
    n = len(cpr_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(cpr_width[i]):
            window = cpr_width[i-lookback+1:i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= lookback // 2:
                # Calculate percentile rank
                count_below = np.sum(valid_window[:-1] < cpr_width[i])
                percentile[i] = count_below / max(len(valid_window[:-1]), 1) * 100.0
    
    return percentile

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    hours = (prices['open_time'].values // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d, bc_1d, tc_1d, cpr_width_1d = calculate_cpr_levels(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    cpr_width_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d)
    
    # Calculate CPR width percentile (narrow CPR = breakout signal)
    cpr_percentile = calculate_cpr_width_percentile(cpr_width_aligned, lookback=30)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Get session hours
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track CPR narrow periods for breakout potential
    narrow_cpr = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(tc_aligned[i]) or np.isnan(bc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === CPR LEVELS ===
        price_above_tc = close[i] > tc_aligned[i]
        price_below_bc = close[i] < bc_aligned[i]
        price_in_cpr = (close[i] >= bc_aligned[i]) and (close[i] <= tc_aligned[i])
        
        # === CPR WIDTH (NARROW = BREAKOUT POTENTIAL) ===
        narrow_cpr = False
        if not np.isnan(cpr_percentile[i]):
            narrow_cpr = cpr_percentile[i] < 30.0  # Bottom 30% = narrow
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        preferred_session = (hours[i] >= 0) and (hours[i] < 12)
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi_7[i] < 40.0
        rsi_overbought = rsi_7[i] > 60.0
        rsi_neutral = (rsi_7[i] >= 40.0) and (rsi_7[i] <= 60.0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (multiple conditions for confluence)
        long_score = 0
        
        # Condition 1: 4h trend bullish
        if htf_bull:
            long_score += 2
        
        # Condition 2: Price above TC (breakout) OR in CPR near BC (pullback)
        if price_above_tc:
            long_score += 2
        elif price_in_cpr and close[i] > bc_aligned[i]:
            long_score += 1
        
        # Condition 3: RSI pullback (oversold in uptrend)
        if rsi_oversold:
            long_score += 2
        elif rsi_neutral:
            long_score += 1
        
        # Condition 4: Narrow CPR (breakout potential)
        if narrow_cpr:
            long_score += 1
        
        # Condition 5: Preferred session
        if preferred_session:
            long_score += 1
        
        # LONG entry threshold (need score >= 5 for entry)
        if long_score >= 5:
            if long_score >= 7:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT entries
        short_score = 0
        
        # Condition 1: 4h trend bearish
        if htf_bear:
            short_score += 2
        
        # Condition 2: Price below BC (breakdown) OR in CPR near TC (pullback)
        if price_below_bc:
            short_score += 2
        elif price_in_cpr and close[i] < tc_aligned[i]:
            short_score += 1
        
        # Condition 3: RSI pullback (overbought in downtrend)
        if rsi_overbought:
            short_score += 2
        elif rsi_neutral:
            short_score += 1
        
        # Condition 4: Narrow CPR (breakout potential)
        if narrow_cpr:
            short_score += 1
        
        # Condition 5: Preferred session
        if preferred_session:
            short_score += 1
        
        # SHORT entry threshold (need score >= 5 for entry)
        if short_score >= 5:
            if short_score >= 7:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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