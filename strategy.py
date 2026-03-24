#!/usr/bin/env python3
"""
Experiment #225: 15m Primary + 4h/1d HTF — Daily CPR Breakout + RSI Pullback

Hypothesis: 15m timeframe with Daily CPR (Central Pivot Range) levels provides 
clear support/resistance for intraday mean reversion. Combined with 4h HMA trend 
bias and relaxed RSI thresholds (35/65 vs 30/70), this should generate 40-100 
trades/year while maintaining quality.

Key innovations vs failed 15m attempts (#217, #221):
- RSI(7) thresholds: 35/65 (not 30/70) → more trades triggered
- Daily CPR: BC/TC from 1d data for S/R levels
- Session filter: UTC 00-12 only (London+NY overlap)
- 4h HMA(21) for trend bias (not too many filters)
- Position size: 0.20 (appropriate for 15m frequency)

Daily CPR Calculation:
- Pivot = (High + Low + Close) / 3
- BC (Bottom Central) = (High + Low) / 2
- TC (Top Central) = (Pivot - BC) + Pivot
- Narrow CPR = (TC - BC) / Pivot < 0.01 → breakout day likely

Entry Logic:
- Long: 4h HMA bull + price > TC + RSI(7) > 45 + session 00-12 UTC
- Short: 4h HMA bear + price < BC + RSI(7) < 55 + session 00-12 UTC
- Pullback: RSI(7) < 40 in uptrend / RSI(7) > 60 in downtrend

Target: Sharpe>0.40, DD>-30%, trades>=30 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_pullback_4h1d_session_v1"
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

def calculate_daily_cpr_from_htf(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d data
    Returns: pivot, bc (bottom central), tc (top central) arrays
    """
    n = len(df_1d)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    
    return pivot, bc, tc

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d, bc_1d, tc_1d = calculate_daily_cpr_from_htf(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 1d HMA for major trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% position size for 15m frequency
    SIZE_STRONG = 0.25  # 25% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only - London+NY overlap) ===
        hour = get_session_hour(open_time[i])
        in_session = (hour >= 0 and hour < 12)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        htf_1d_bear = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === DAILY CPR LEVELS ===
        tc = tc_aligned[i]
        bc = bc_aligned[i]
        pivot = pivot_aligned[i]
        
        # Narrow CPR detection (breakout day likely)
        cpr_width = (tc - bc) / pivot if pivot > 1e-10 and not np.isnan(tc) and not np.isnan(bc) else 1.0
        narrow_cpr = cpr_width < 0.015
        
        # Price position relative to CPR
        above_tc = close[i] > tc if not np.isnan(tc) else False
        below_bc = close[i] < bc if not np.isnan(bc) else False
        inside_cpr = (close[i] >= bc and close[i] <= tc) if (not np.isnan(bc) and not np.isnan(tc)) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_ma[i] * 1.2 if not np.isnan(vol_ma[i]) else True
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI VALUES (relaxed thresholds for more trades) ===
        rsi_low = rsi_7[i] < 40.0  # Oversold for long pullback
        rsi_high = rsi_7[i] > 60.0  # Overbought for short pullback
        rsi_neutral_long = rsi_7[i] > 45.0  # Not oversold for breakout long
        rsi_neutral_short = rsi_7[i] < 55.0  # Not overbought for breakout short
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if in_session:
            # Long breakout: price > TC + 4h bull + RSI neutral + volume
            if above_tc and htf_4h_bull and rsi_neutral_long and vol_confirmed:
                desired_signal = SIZE_STRONG
            
            # Long pullback: 4h bull + RSI oversold + price > BC (support)
            elif htf_4h_bull and rsi_low and not below_bc:
                desired_signal = SIZE_BASE
            
            # Long CPR bounce: inside CPR + 4h bull + RSI < 50
            elif inside_cpr and htf_4h_bull and rsi_7[i] < 50.0:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES
        if in_session:
            # Short breakdown: price < BC + 4h bear + RSI neutral + volume
            if below_bc and htf_4h_bear and rsi_neutral_short and vol_confirmed:
                desired_signal = -SIZE_STRONG
            
            # Short pullback: 4h bear + RSI overbought + price < TC (resistance)
            elif htf_4h_bear and rsi_high and not above_tc:
                desired_signal = -SIZE_BASE
            
            # Short CPR rejection: inside CPR + 4h bear + RSI > 50
            elif inside_cpr and htf_4h_bear and rsi_7[i] > 50.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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