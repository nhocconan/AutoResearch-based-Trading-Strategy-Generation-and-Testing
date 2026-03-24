#!/usr/bin/env python3
"""
Experiment #581: 15m Primary + 1h/4h/1d HTF — Daily Pivot + HMA Trend + RSI Pullback

Hypothesis: 15m timeframe with daily pivot levels (CPR from 1d) provides superior
intraday entry points when aligned with HTF trend. Daily pivots act as natural
support/resistance that institutions watch. Combined with 4h HMA trend bias and
15m RSI pullback entries, this captures trend continuations at key levels.

Key innovations for 15m (FIRST 15m experiment):
1. Daily CPR (Central Pivot Range) from 1d HTF - BC/TC/CP levels
2. 4h HMA(21) for medium trend bias
3. 1h HMA(21) for short-term trend alignment
4. 15m RSI(7) for pullback entries (faster than RSI14)
5. Session filter: 00-12 UTC (London+NY overlap = 70% of crypto volume)
6. Narrow CPR filter: CPR width < 0.5% = breakout day potential
7. ATR(14)*2.0 stoploss on all positions

Strategy logic:
1. 1d CPR calculation: CP = (H+L+C)/3, BC = (H+L)/2, TC = CP + (CP-BC)
2. 4h HMA(21) = medium trend bias (price > HMA = bull)
3. 1h HMA(21) = short-term alignment (must match 4h direction)
4. 15m RSI(7) < 35 = long entry in uptrend, > 65 = short in downtrend
5. Price near CPR level (within 0.5%) = confluence
6. Session filter: only trade 00-12 UTC (high volume hours)

Entry confluence (need 3+):
- HTF trend alignment (4h + 1h HMA same direction)
- Price at/near daily pivot level (CPR BC/TC/CP)
- RSI pullback extreme (RSI7 < 35 or > 65)
- Narrow CPR (width < 0.5% = potential breakout)
- Session filter (00-12 UTC)

Target: Sharpe>0.40, trades>=40 train (10/year), trades>=5 test
Timeframe: 15m
Size: 0.20 base, 0.25 strong (lower size for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pivot_hma_rsi_session_1h4h1d_v1"
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

def calculate_cpr_from_daily(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from daily OHLC
    CP = Central Pivot = (High + Low + Close) / 3
    BC = Bottom Central = (High + Low) / 2
    TC = Top Central = CP + (CP - BC)
    
    Returns arrays aligned to daily bars
    """
    n = len(df_1d)
    cp = np.zeros(n)
    bc = np.zeros(n)
    tc = np.zeros(n)
    cpr_width = np.zeros(n)
    
    for i in range(n):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        
        cp[i] = (h + l + c) / 3.0
        bc[i] = (h + l) / 2.0
        tc[i] = cp[i] + (cp[i] - bc[i])
        
        # CPR width as % of price
        if cp[i] > 1e-10:
            cpr_width[i] = abs(tc[i] - bc[i]) / cp[i] * 100.0
        else:
            cpr_width[i] = 0.0
    
    return cp, bc, tc, cpr_width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for short-term trend
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 4h HMA for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d CPR and align
    cp_1d, bc_1d, tc_1d, cpr_width_1d = calculate_cpr_from_daily(df_1d)
    cp_aligned = align_htf_to_ltf(prices, df_1d, cp_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    cpr_width_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(rsi[i]) or np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(cp_aligned[i]) or np.isnan(bc_aligned[i]) or np.isnan(tc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = (hour_utc >= 0 and hour_utc < 12)  # London+NY overlap
        
        # === HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1h_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # HTF alignment (both 1h and 4h agree)
        htf_aligned_bull = hma_4h_aligned[i] > hma_4h_aligned[i-10] if i >= 10 and not np.isnan(hma_4h_aligned[i-10]) else False
        htf_aligned_bear = hma_4h_aligned[i] < hma_4h_aligned[i-10] if i >= 10 and not np.isnan(hma_4h_aligned[i-10]) else False
        
        # === CPR LEVELS ===
        cpr_narrow = cpr_width_aligned[i] < 0.5 if not np.isnan(cpr_width_aligned[i]) else False
        
        # Price distance to CPR levels
        dist_to_cp = abs(close[i] - cp_aligned[i]) / cp_aligned[i] * 100.0 if cp_aligned[i] > 1e-10 else 100.0
        dist_to_bc = abs(close[i] - bc_aligned[i]) / bc_aligned[i] * 100.0 if bc_aligned[i] > 1e-10 else 100.0
        dist_to_tc = abs(close[i] - tc_aligned[i]) / tc_aligned[i] * 100.0 if tc_aligned[i] > 1e-10 else 100.0
        
        near_cpr = dist_to_cp < 0.5 or dist_to_bc < 0.5 or dist_to_tc < 0.5
        
        # === RSI PULLBACK ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        
        # RSI recovering from extreme
        rsi_recover_long = rsi_oversold and i > 0 and rsi[i] > rsi[i-1]
        rsi_recover_short = rsi_overbought and i > 0 and rsi[i] < rsi[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entries (need 3+ confluence)
        if htf_bull:
            confluence_count = 0
            if htf_aligned_bull:
                confluence_count += 1
            if near_cpr:
                confluence_count += 1
            if rsi_oversold or rsi_recover_long:
                confluence_count += 1
            if is_session:
                confluence_count += 1
            if cpr_narrow:
                confluence_count += 1
            
            if confluence_count >= 3:
                if rsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                elif rsi_oversold:
                    desired_signal = SIZE_BASE
        
        # SHORT entries (need 3+ confluence)
        elif htf_bear:
            confluence_count = 0
            if htf_aligned_bear:
                confluence_count += 1
            if near_cpr:
                confluence_count += 1
            if rsi_overbought or rsi_recover_short:
                confluence_count += 1
            if is_session:
                confluence_count += 1
            if cpr_narrow:
                confluence_count += 1
            
            if confluence_count >= 3:
                if rsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                elif rsi_overbought:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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