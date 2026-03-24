#!/usr/bin/env python3
"""
Experiment #025: 15m Primary + 4h/1d HTF — Daily CPR Breakout + HMA Trend + RSI Pullback

Hypothesis: 15m strategies fail due to too many trades → fee drag. Solution:
- Use Daily CPR (Central Pivot Range) from 1d HTF as key S/R levels
- Use 4h HMA for strong trend bias (only trade in direction)
- Use 15m RSI(7) for pullback entries (not breakouts - less whipsaw)
- Session filter: 00-12 UTC (London/NY overlap = highest volume)
- Volume confirmation: must be above 20-bar average
- This creates HIGH-SELECTIVITY entries (4-5 confluence factors)
- Target: 40-100 trades/year, Sharpe>0.5, DD<-30%

Key design choices:
- Timeframe: 15m (higher frequency but VERY selective entries)
- HTF: 1d CPR levels + 4h HMA trend
- Entry: RSI(7) pullback to CPR level + HTF trend + session + volume
- Position size: 0.20 (smaller for 15m frequency)
- Stoploss: 2.0x ATR trailing
- Discrete signals: 0.0, ±0.20 only (minimize churn)

Target: Beat mtf_12h_kama_adx_chop_regime_1d_v1 (Sharpe=0.019)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_hma_rsi_session_4h1d_v1"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_cpr_from_1d(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d data
    CPR = [BC, Pivot, TC]
    BC = (High + Low) / 2
    Pivot = (High + Low + Close) / 3
    TC = (Pivot - BC) + Pivot
    Narrow CPR = TC - BC < threshold (breakout potential)
    """
    n = len(df_1d)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    bc = (high + low) / 2.0
    pivot = (high + low + close) / 3.0
    tc = (pivot - bc) + pivot
    cpr_width = tc - bc
    
    return bc, pivot, tc, cpr_width

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
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    bc_1d_raw, pivot_1d_raw, tc_1d_raw, cpr_width_1d_raw = calculate_cpr_from_1d(df_1d)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    cpr_width_1d_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=13)
    rsi_fast = calculate_rsi(close, period=7)  # Fast RSI for entry timing
    rsi_std = calculate_rsi(close, period=14)  # Standard RSI for filter
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_std[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        # Parse hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_active_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m TREND (HMA13) ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === CPR LEVELS (from 1d HTF) ===
        bc = bc_1d_aligned[i]
        pivot = pivot_1d_aligned[i]
        tc = tc_1d_aligned[i]
        cpr_width = cpr_width_1d_aligned[i]
        
        # Narrow CPR = breakout potential (width < 1% of price)
        narrow_cpr = cpr_width < (pivot * 0.01) if not np.isnan(cpr_width) else False
        
        # Price position relative to CPR
        price_above_tc = close[i] > tc
        price_below_bc = close[i] < bc
        price_near_pivot = abs(close[i] - pivot) < (pivot * 0.005)  # within 0.5%
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI(7) oversold (<35) in uptrend
        rsi_oversold = rsi_fast[i] < 35.0
        # Short: RSI(7) overbought (>65) in downtrend
        rsi_overbought = rsi_fast[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > (vol_avg[i] * 1.2)  # 20% above average
        
        # === DESIRED SIGNAL (Multi-Confluence Logic) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bull + 15m bull + RSI pullback + volume + session
        # Must be above CPR pivot or breaking above TC (narrow CPR)
        long_conditions = (
            htf_bull and  # 4h trend up
            hma_bull and  # 15m trend up
            rsi_oversold and  # RSI pullback
            volume_confirmed and  # Volume confirmation
            is_active_session and  # Active session
            (price_above_tc or (narrow_cpr and price_near_pivot))  # CPR breakout or narrow
        )
        
        # SHORT ENTRY: 4h bear + 15m bear + RSI pullback + volume + session
        # Must be below CPR pivot or breaking below BC (narrow CPR)
        short_conditions = (
            htf_bear and  # 4h trend down
            hma_bear and  # 15m trend down
            rsi_overbought and  # RSI pullback
            volume_confirmed and  # Volume confirmation
            is_active_session and  # Active session
            (price_below_bc or (narrow_cpr and price_near_pivot))  # CPR breakdown or narrow
        )
        
        if long_conditions:
            desired_signal = SIZE
        elif short_conditions:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
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