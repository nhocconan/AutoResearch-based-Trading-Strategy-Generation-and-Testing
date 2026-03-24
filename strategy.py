#!/usr/bin/env python3
"""
Experiment #505: 15m Primary + 4h/1d HTF — Pivot Boss CPR Breakout

Hypothesis: 15m timeframe needs VERY selective entries to avoid fee drag.
Using Daily CPR (Central Pivot Range) from 1d HTF + 4h HMA trend + 15m RSI timing.
Session filter (00-12 UTC) reduces trades to target 50-80/year.

Strategy logic:
1. 1d CPR (BC/TC/CP) = daily support/resistance levels from HTF
2. 4h HMA(21) = intraday trend bias (faster than 1d for 15m entries)
3. 15m RSI(7) = entry timing (oversold <25 long, overbought >75 short)
4. Session filter: only trade 00-12 UTC (London+NY overlap, highest liquidity)
5. Narrow CPR filter: CPR width < 0.5% = consolidation before breakout
6. ATR(14)*2.0 stoploss on all positions
7. OR logic for entries (CPR breakout OR RSI extreme + trend)

Key for 15m success:
- SMALL position size: 0.15-0.20 (higher frequency = smaller size)
- Session filter cuts trades by ~60%
- 3+ confluence: 4h trend + CPR level + RSI timing
- Target 50-80 trades/year (not 300+ which kills PnL with fees)

Why this might work on 15m:
- Prior 15m failures (#497, #501) had Sharpe=0.000 (0 trades = too strict)
- This uses LOOSE RSI (25/75 not 20/80) + OR logic for entries
- CPR breakout is proven on lower timeframes (Pivot Boss methodology)
- Session filter ensures quality over quantity

Target: Sharpe>0.40, trades>=150 train (40/year), trades>=25 test
Timeframe: 15m (first proper 15m with correct MTF + session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_session_4h1d_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_cpr_levels(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d data.
    CPR = (BC, CP, TC) where:
    - CP (Central Pivot) = (High + Low + Close) / 3
    - BC (Bottom Central) = (High + Low) / 2
    - TC (Top Central) = 2*CP - BC
    
    Returns arrays aligned to 1d bars.
    """
    n = len(df_1d)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    cp = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = 2.0 * cp - bc
    
    return bc, cp, tc

def calculate_session_hour(prices):
    """Extract UTC hour from open_time for session filtering"""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000.0
    hours = (timestamps % 86400) / 3600.0  # Convert to hour of day (0-23)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intraday trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    bc_1d_raw, cp_1d_raw, tc_1d_raw = calculate_cpr_levels(df_1d)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    cp_1d_aligned = align_htf_to_ltf(prices, df_1d, cp_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 1d SMA for additional trend filter
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m timing
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Session hours (UTC)
    session_hours = calculate_session_hour(prices)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        in_session = session_hours[i] >= 0.0 and session_hours[i] <= 12.0
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1d CPR LEVELS ===
        cpr_width = (tc_1d_aligned[i] - bc_1d_aligned[i]) / cp_1d_aligned[i] if cp_1d_aligned[i] > 0 else 1.0
        narrow_cpr = cpr_width < 0.005  # CPR width < 0.5% = consolidation
        
        price_above_cpr = close[i] > tc_1d_aligned[i]
        price_below_cpr = close[i] < bc_1d_aligned[i]
        price_in_cpr = (close[i] >= bc_1d_aligned[i]) and (close[i] <= tc_1d_aligned[i])
        
        # === 1d TREND FILTER ===
        above_sma1d = not np.isnan(sma_1d_aligned[i]) and close[i] > sma_1d_aligned[i]
        below_sma1d = not np.isnan(sma_1d_aligned[i]) and close[i] < sma_1d_aligned[i]
        
        # === 15m RSI EXTREMES (LOOSE: 25/75 for 15m) ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 else False
        
        # === 15m HMA CROSSOVER ===
        hma_15m_prev = hma_15m[i-1] if i > 0 else np.nan
        hma_bull_cross = (close[i] > hma_15m[i]) and (not np.isnan(hma_15m_prev)) and (close[i-1] <= hma_15m_prev)
        hma_bear_cross = (close[i] < hma_15m[i]) and (not np.isnan(hma_15m_prev)) and (close[i-1] >= hma_15m_prev)
        
        # === 15m TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 2.5
        
        # === ENTRY LOGIC (OR logic for more trades) ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if in_session and vol_normal:
            # === LONG ENTRIES ===
            if htf_bull or above_sma1d:  # 4h or 1d trend bull
                # CPR breakout long (price breaks above TC)
                if price_above_cpr and narrow_cpr:
                    desired_signal = SIZE_STRONG
                # RSI oversold bounce in uptrend
                elif rsi_extreme_oversold and rsi_rising and hma_15m_bull:
                    desired_signal = SIZE_BASE
                # HMA crossover long
                elif hma_bull_cross and close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False:
                    desired_signal = SIZE_BASE
                # RSI recovery from oversold
                elif rsi_7[i] > 35.0 and rsi_7[i-1] <= 35.0 and htf_bull:
                    desired_signal = SIZE_BASE * 0.8
            
            # === SHORT ENTRIES ===
            elif htf_bear or below_sma1d:  # 4h or 1d trend bear
                # CPR breakdown short (price breaks below BC)
                if price_below_cpr and narrow_cpr:
                    desired_signal = -SIZE_STRONG
                # RSI overbought rejection in downtrend
                elif rsi_extreme_overbought and rsi_falling and hma_15m_bear:
                    desired_signal = -SIZE_BASE
                # HMA crossover short
                elif hma_bear_cross and close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False:
                    desired_signal = -SIZE_BASE
                # RSI breakdown from overbought
                elif rsi_7[i] < 65.0 and rsi_7[i-1] >= 65.0 and htf_bear:
                    desired_signal = -SIZE_BASE * 0.8
            
            # === MEAN REVERSION (works in any regime) ===
            if desired_signal == 0.0:
                # Long at CPR support (BC) with RSI oversold
                if price_in_cpr and close[i] <= bc_1d_aligned[i] * 1.002 and rsi_extreme_oversold:
                    desired_signal = SIZE_BASE * 0.8
                # Short at CPR resistance (TC) with RSI overbought
                elif price_in_cpr and close[i] >= tc_1d_aligned[i] * 0.998 and rsi_extreme_overbought:
                    desired_signal = -SIZE_BASE * 0.8
        
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