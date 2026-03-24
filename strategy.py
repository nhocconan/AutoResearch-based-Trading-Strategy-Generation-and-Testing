#!/usr/bin/env python3
"""
Experiment #297: 15m Primary + 4h/1d HTF — Daily CPR Breakout + RSI Pullback v1

Hypothesis: 15m timeframe needs VERY selective entries (40-100 trades/year) to avoid fee drag.
Using Daily CPR (Central Pivot Range) from 1d HTF as key S/R levels, combined with
4h trend direction and 15m RSI(7) pullback entries. This is proven in traditional
trading (Pivot Boss methodology) but untested on crypto 15m.

Key innovations:
1. DAILY CPR LEVELS: BC/TC/CP from 1d HTF act as magnet zones for entries
2. NARROW CPR FILTER: Only trade when CPR width < 1.5% of price (consolidation)
3. 4H HMA TREND: Direction filter - only long if 4h HMA bullish, vice versa
4. RSI(7) PULLBACK: Enter on RSI(7) < 35 (long) or > 65 (short) within trend
5. VOLUME CONFIRMATION: Entry bar volume > 1.3x 20-bar avg (avoids fakeouts)
6. SESSION FILTER: Prefer 00-12 UTC (London+NY overlap for crypto liquidity)

Position sizing: 0.15 base, 0.25 when all confluence align (discrete levels)
Stoploss: 2.5x ATR(14) from entry price
Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols positive

CRITICAL: 15m generates many bars - must be VERY selective to avoid >100 trades/year.
Use 4+ confluence: HTF trend + CPR level + RSI extreme + volume spike.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_pullback_volume_4h1d_v1"
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

def calculate_cpr_from_1d(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) from 1d data
    CP = (High + Low + Close) / 3
    BC = (High + Low) / 2
    TC = (CP - BC) + CP
    
    Returns arrays aligned to 1d bars
    """
    n = len(df_1d)
    cp = np.zeros(n)
    bc = np.zeros(n)
    tc = np.zeros(n)
    cp[:] = np.nan
    bc[:] = np.nan
    tc[:] = np.nan
    
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    for i in range(1, n):
        cp[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        bc[i] = (high[i-1] + low[i-1]) / 2.0
        tc[i] = (cp[i] - bc[i]) + cp[i]
    
    return cp, bc, tc

def calculate_cpr_width(cp, bc, tc):
    """
    CPR Width = (TC - BC) / CP * 100
    Narrow CPR (< 1.5%) indicates consolidation -> breakout likely
    """
    n = len(cp)
    width = np.zeros(n)
    width[:] = np.nan
    
    for i in range(n):
        if not np.isnan(cp[i]) and cp[i] > 1e-10:
            width[i] = (tc[i] - bc[i]) / cp[i] * 100.0
    
    return width

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    hours = np.zeros(len(prices), dtype=int)
    for i in range(len(prices)):
        try:
            # open_time is in milliseconds
            ts = prices['open_time'].iloc[i] / 1000.0
            import datetime
            dt = datetime.datetime.utcfromtimestamp(ts)
            hours[i] = dt.hour
        except:
            hours[i] = 0
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    cp_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr_from_1d(df_1d)
    cp_1d_aligned = align_htf_to_ltf(prices, df_1d, cp_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate CPR width (narrow = consolidation)
    cpr_width_raw = calculate_cpr_width(cp_1d_raw, bc_1d_raw, tc_1d_raw)
    cpr_width_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume moving average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (UTC)
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(cp_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4H TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15M HMA TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === CPR LEVELS ===
        cp = cp_1d_aligned[i]
        bc = bc_1d_aligned[i]
        tc = tc_1d_aligned[i]
        cpr_w = cpr_width_aligned[i]
        
        # Narrow CPR filter (< 1.5% = consolidation, breakout likely)
        narrow_cpr = not np.isnan(cpr_w) and cpr_w < 1.5
        
        # Price position relative to CPR
        price_above_cpr = close[i] > tc
        price_below_cpr = close[i] < bc
        price_in_cpr = bc <= close[i] <= tc
        
        # === RSI CONDITIONS (7-period for faster response) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_neutral = 40.0 <= rsi_7[i] <= 60.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = volume[i] > 1.3 * vol_sma_20[i] if vol_sma_20[i] > 1e-10 else False
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        preferred_session = 0 <= hours[i] <= 12
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (4+ confluence required) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG SETUP
        long_conditions = []
        
        # 1. 4h trend bullish
        if htf_4h_bull:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # 2. 15m HMA bullish OR price above SMA50
        if hma_15m_bull or close[i] > sma_50[i]:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # 3. RSI oversold (pullback entry)
        if rsi_oversold:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # 4. Price near CPR support (at or above BC)
        if close[i] >= bc * 0.995 or price_above_cpr:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # 5. Volume spike confirmation
        if vol_spike:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # 6. Above SMA200 (major trend)
        if above_sma200:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        confluence_long = sum(long_conditions)
        
        # SHORT SETUP
        short_conditions = []
        
        # 1. 4h trend bearish
        if htf_4h_bear:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # 2. 15m HMA bearish OR price below SMA50
        if hma_15m_bear or close[i] < sma_50[i]:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # 3. RSI overbought (pullback entry)
        if rsi_overbought:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # 4. Price near CPR resistance (at or below TC)
        if close[i] <= tc * 1.005 or price_below_cpr:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # 5. Volume spike confirmation
        if vol_spike:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # 6. Below SMA200 (major trend)
        if below_sma200:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        confluence_short = sum(short_conditions)
        
        # Require 4+ confluence for entry (very selective for 15m)
        if confluence_long >= 4:
            # Boost size if narrow CPR + preferred session
            if narrow_cpr and preferred_session:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        elif confluence_short >= 4:
            if narrow_cpr and preferred_session:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
        
        signals[i] = final_signal
    
    return signals