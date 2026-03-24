#!/usr/bin/env python3
"""
Experiment #205: 15m Primary + 4h/1d HTF — Daily CPR Breakout + Session Filter

Hypothesis: 15m timeframe with Daily CPR (Central Pivot Range) levels from 1d HTF
can capture institutional breakout moves during high-volume sessions. Previous 15m
attempts failed due to too many filters (0 trades). This version simplifies:

Daily CPR Components (from 1d HTF):
- TC (Top Central) = (H + L + C) / 3
- BC (Bottom Central) = (H + L) / 2
- Pivot = (H + L + C) / 3
- Narrow CPR = when TC-BC width < 20% of ATR → breakout likely

Entry Logic:
- Long: 15m close > TC + 4h HMA bullish + RSI(7) > 50 + session 00-12 UTC
- Short: 15m close < BC + 4h HMA bearish + RSI(7) < 50 + session 00-12 UTC
- Narrow CPR breakout gets 1.5x position size (higher conviction)

Session Filter:
- 00-12 UTC = London + NY overlap (highest crypto volume)
- Reduces trade frequency to 50-80/year target

4h HTF Filter:
- HMA(21) for trend direction
- Only trade breakouts in direction of 4h trend

Position sizing: 0.20 base, 0.30 for narrow CPR breakouts
Stoploss: 2.0x ATR trailing
Target: Sharpe>0.40, DD>-35%, trades>=40 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_session_breakout_4h1d_v1"
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

def calculate_daily_cpr(df_daily):
    """
    Calculate Daily CPR (Central Pivot Range) levels
    TC = Top Central, BC = Bottom Central, Pivot = standard pivot
    
    Returns arrays aligned with daily bars
    """
    n = len(df_daily)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    high = df_daily['high'].values
    low = df_daily['low'].values
    close = df_daily['close'].values
    
    tc = np.zeros(n)
    bc = np.zeros(n)
    pivot = np.zeros(n)
    cpr_width = np.zeros(n)
    
    tc[:] = np.nan
    bc[:] = np.nan
    pivot[:] = np.nan
    cpr_width[:] = np.nan
    
    for i in range(1, n):
        # Use PREVIOUS day's data for current day's levels (no look-ahead)
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        
        pivot[i] = (h + l + c) / 3.0
        tc[i] = (h + l + c) / 3.0  # Same as pivot in standard CPR
        bc[i] = (h + l) / 2.0
        cpr_width[i] = tc[i] - bc[i]
    
    return tc, bc, pivot, cpr_width

def is_session_active(open_time_unix, start_hour=0, end_hour=12):
    """
    Check if timestamp is within session hours (UTC)
    Default: 00-12 UTC (London + NY overlap for crypto)
    """
    # Convert unix ms to hour
    hour = (open_time_unix // (1000 * 60 * 60)) % 24
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 1d CPR levels
    tc_1d_raw, bc_1d_raw, pivot_1d_raw, cpr_width_1d_raw = calculate_daily_cpr(df_1d)
    tc_1d = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    bc_1d = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    cpr_width_1d = align_htf_to_ltf(prices, df_1d, cpr_width_1d_raw)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d ATR for CPR width comparison
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% base position size for 15m
    SIZE_STRONG = 0.30  # 30% for narrow CPR breakouts
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(tc_1d[i]) or np.isnan(bc_1d[i]):
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
        
        # === SESSION FILTER (00-12 UTC) ===
        in_session = is_session_active(open_time[i], start_hour=0, end_hour=12)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h[i]
        htf_4h_bear = close[i] < hma_4h[i]
        
        # === CPR LEVELS ===
        tc = tc_1d[i]
        bc = bc_1d[i]
        cpr_w = cpr_width_1d[i]
        atr_daily = atr_1d[i] if not np.isnan(atr_1d[i]) else atr[i] * 96  # Approximate daily ATR
        
        # === NARROW CPR DETECTION ===
        # CPR width < 20% of daily ATR = narrow = breakout likely
        is_narrow_cpr = False
        if not np.isnan(cpr_w) and not np.isnan(atr_daily) and atr_daily > 1e-10:
            is_narrow_cpr = cpr_w < 0.20 * atr_daily
        
        # === PRICE POSITION RELATIVE TO CPR ===
        above_tc = close[i] > tc
        below_bc = close[i] < bc
        inside_cpr = (close[i] >= bc) and (close[i] <= tc)
        
        # === BREAKOUT CONFIRMATION ===
        # Price broke above TC from inside/below
        breakout_long = above_tc and (close[i-1] <= tc if not np.isnan(tc) else False)
        # Price broke below BC from inside/above
        breakout_short = below_bc and (close[i-1] >= bc if not np.isnan(bc) else False)
        
        # === MOMENTUM CONFIRMATION ===
        rsi_bull = rsi_7[i] > 50.0
        rsi_bear = rsi_7[i] < 50.0
        rsi_extreme_long = rsi_7[i] < 35.0  # Oversold bounce setup
        rsi_extreme_short = rsi_7[i] > 65.0  # Overbought fade setup
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if in_session:
            # Breakout long with trend + momentum
            if above_tc and htf_4h_bull and rsi_bull:
                if is_narrow_cpr:
                    desired_signal = SIZE_STRONG  # High conviction narrow CPR
                else:
                    desired_signal = SIZE_BASE
            
            # Mean reversion long inside CPR (fade to BC)
            elif inside_cpr and rsi_extreme_long and htf_4h_bull:
                desired_signal = SIZE_BASE * 0.7
        
        # SHORT ENTRIES
        if in_session:
            # Breakout short with trend + momentum
            if below_bc and htf_4h_bear and rsi_bear:
                if is_narrow_cpr:
                    desired_signal = -SIZE_STRONG  # High conviction narrow CPR
                else:
                    desired_signal = -SIZE_BASE
            
            # Mean reversion short inside CPR (fade to TC)
            elif inside_cpr and rsi_extreme_short and htf_4h_bear:
                desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
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
            final_signal = SIZE_BASE * 0.7
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals