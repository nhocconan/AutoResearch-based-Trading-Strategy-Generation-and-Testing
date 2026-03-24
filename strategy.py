#!/usr/bin/env python3
"""
Experiment #709: 15m Primary + 1h/4h/1d HTF — Multi-TF Confluence with Session Filter

Hypothesis: 15m strategies fail due to too many trades and fee drag. Solution:
1. Use 1d/4h HMA for TREND DIRECTION (only trade with HTF alignment)
2. Use 1h HMA for INTERMEDIATE confirmation
3. Use 15m RSI(7) + Volume for ENTRY TIMING only
4. Session filter: UTC 00-12 (London/NY overlap) - avoid Asian chop
5. Daily CPR levels from 1d for support/resistance confluence
6. VERY strict entry: need 4+ conditions aligned
7. Small position size (0.15-0.25) due to higher frequency

Key innovations:
1. Triple HTF alignment (1d + 4h + 1h HMA) before any 15m entry
2. Volume confirmation (15m volume > 1.5x 20-bar average)
3. Session filter for high-liquidity hours only
4. RSI(7) pullback entry (not extreme values - RSI 25-40 long, 60-75 short)
5. ATR(14) 2.5x trailing stop with signal→0 on breach
6. Discrete sizing: 0.0, ±0.15, ±0.25 to minimize fee churn

Target: Sharpe>0.40, trades=40-100/year, DD>-30%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller than 6h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_session_cpr_1h4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_daily_cpr(df_1d):
    """
    Calculate Daily Central Pivot Range (CPR) from 1d data
    CPR = (Pivot, BC, TC) where:
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    """
    n = len(df_1d)
    pivot = np.full(n, np.nan)
    bc = np.full(n, np.nan)
    tc = np.full(n, np.nan)
    
    for i in range(n):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        
        pivot[i] = (h + l + c) / 3.0
        bc[i] = (h + l) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
    
    return pivot, bc, tc

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    utc_hour = (ts_seconds % 86400) / 3600.0
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Daily CPR and align
    pivot_1d, bc_1d, tc_1d = calculate_daily_cpr(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 15m indicators
    hma_15m_21 = calculate_hma(close, period=21)
    hma_15m_50 = calculate_hma(close, period=50)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    SIZE_WEAK = 0.10
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_21[i]) or np.isnan(hma_15m_50[i]):
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
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_aligned[i]) or np.isnan(bc_aligned[i]) or np.isnan(tc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only - London/NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        session_active = (utc_hour >= 0 and utc_hour < 12)
        
        # === HTF TREND BIAS (Triple alignment required) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # Full HTF alignment (all 3 timeframes agree)
        htf_full_bull = htf_1d_bull and htf_4h_bull and htf_1h_bull
        htf_full_bear = htf_1d_bear and htf_4h_bear and htf_1h_bear
        
        # Partial HTF alignment (2 of 3 agree)
        htf_partial_bull = (htf_1d_bull + htf_4h_bull + htf_1h_bull) >= 2
        htf_partial_bear = (htf_1d_bear + htf_4h_bear + htf_1h_bear) >= 2
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_15m_21[i] > hma_15m_50[i]
        hma_15m_bear = hma_15m_21[i] < hma_15m_50[i]
        
        # === RSI PULLBACK ENTRY (not extremes - pullback in trend) ===
        # Long: RSI(7) pulled back to 25-40 in uptrend
        rsi_long_pullback = (rsi_7[i] >= 25.0 and rsi_7[i] <= 40.0)
        # Short: RSI(7) rallied to 60-75 in downtrend
        rsi_short_pullback = (rsi_7[i] >= 60.0 and rsi_7[i] <= 75.0)
        
        # === VOLUME CONFIRMATION ===
        vol_confirmation = volume[i] > 1.5 * vol_sma[i]
        
        # === CPR LEVEL CONFLUENCE ===
        # Long: price near or above CPR (support)
        cpr_long_support = close[i] >= bc_aligned[i]
        # Short: price near or below CPR (resistance)
        cpr_short_resist = close[i] <= tc_aligned[i]
        
        # === ENTRY LOGIC (VERY STRICT - 4+ confluence required) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG entry conditions
        if session_active:
            confluence_count = 0
            
            if htf_full_bull:
                confluence_count += 2
            elif htf_partial_bull:
                confluence_count += 1
            
            if hma_15m_bull:
                confluence_count += 1
            
            if rsi_long_pullback:
                confluence_count += 1
            
            if vol_confirmation:
                confluence_count += 1
            
            if cpr_long_support:
                confluence_count += 1
            
            # Need 4+ confluence for entry
            if confluence_count >= 4 and htf_full_bull:
                desired_signal = SIZE_STRONG
            elif confluence_count >= 4 and htf_partial_bull and hma_15m_bull:
                desired_signal = SIZE_BASE
            elif confluence_count >= 3 and htf_full_bull and rsi_long_pullback:
                desired_signal = SIZE_WEAK
        
        # SHORT entry conditions
        if session_active:
            confluence_count = 0
            
            if htf_full_bear:
                confluence_count += 2
            elif htf_partial_bear:
                confluence_count += 1
            
            if hma_15m_bear:
                confluence_count += 1
            
            if rsi_short_pullback:
                confluence_count += 1
            
            if vol_confirmation:
                confluence_count += 1
            
            if cpr_short_resist:
                confluence_count += 1
            
            # Need 4+ confluence for entry
            if confluence_count >= 4 and htf_full_bear:
                desired_signal = -SIZE_STRONG
            elif confluence_count >= 4 and htf_partial_bear and hma_15m_bear:
                desired_signal = -SIZE_BASE
            elif confluence_count >= 3 and htf_full_bear and rsi_short_pullback:
                desired_signal = -SIZE_WEAK
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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