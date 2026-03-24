#!/usr/bin/env python3
"""
Experiment #605: 15m Primary + 4h/1d HTF — Camarilla Pivot + Session + Volume Confluence

Hypothesis: 15m timeframe with strict confluence filters can achieve high Sharpe by:
1. Using 4h HMA for trend bias (only trade in HTF direction)
2. Daily Camarilla pivot levels (R3/S3 for mean-reversion, R4/S4 for breakout)
3. Session filter (00-12 UTC) to avoid low-volume Asian session
4. Volume spike confirmation (vol > 1.5x 20-period avg)
5. RSI(7) extremes for 15m entry timing
6. Choppiness Index for regime detection (trend vs range)

Key innovations vs failed 15m experiments (#597, #601):
- Added session filter to reduce trades by ~40%
- Volume confirmation filter (avoids fake breakouts)
- Camarilla pivots from 1d HTF (proven S/R levels)
- Smaller position size (0.18 vs 0.30) for higher frequency
- 3+ confluence required for entry (HTF + pivot + volume + RSI)

Target: 40-80 trades/year, Sharpe>0.40, DD<-25%
Timeframe: 15m
Size: 0.15-0.20 (discrete levels to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_session_vol_4h1d_v1"
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

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Points - tighter levels than standard pivots
    R3/S3 = mean reversion levels
    R4/S4 = breakout levels
    
    H = (High + Low + Close) / 3
    Range = High - Low
    
    R4 = Close + (Range * 1.5000)
    R3 = Close + (Range * 1.2500)
    R2 = Close + (Range * 1.1666)
    R1 = Close + (Range * 1.0833)
    
    S4 = Close - (Range * 1.5000)
    S3 = Close - (Range * 1.2500)
    S2 = Close - (Range * 1.1666)
    S1 = Close - (Range * 1.0833)
    """
    hlc3 = (high + low + close) / 3.0
    range_val = high - low
    
    r4 = prev_close + (range_val * 1.5000)
    r3 = prev_close + (range_val * 1.2500)
    r2 = prev_close + (range_val * 1.1666)
    r1 = prev_close + (range_val * 1.0833)
    
    s4 = prev_close - (range_val * 1.5000)
    s3 = prev_close - (range_val * 1.2500)
    s2 = prev_close - (range_val * 1.1666)
    s1 = prev_close - (range_val * 1.0833)
    
    return r1, r2, r3, r4, s1, s2, s3, s4

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
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1d Camarilla pivots (use previous day's OHLC)
    daily_h = df_1d['high'].values
    daily_l = df_1d['low'].values
    daily_c = df_1d['close'].values
    daily_c_prev = np.roll(daily_c, 1)
    daily_c_prev[0] = daily_c[0]
    
    daily_r1, daily_r2, daily_r3, daily_r4, daily_s1, daily_s2, daily_s3, daily_s4 = \
        calculate_camarilla_pivots(daily_h, daily_l, daily_c, daily_c_prev)
    
    # Align daily pivots to 15m
    r3_aligned = align_htf_to_ltf(prices, df_1d, daily_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, daily_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, daily_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, daily_s4)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA for spike detection
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.22
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === HTF BIAS (4h trend + 1d macro) ===
        htf_bull = close[i] > hma_4h_aligned[i] and close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] and close[i] < hma_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === VOLUME SPIKE FILTER ===
        vol_spike = volume[i] > 1.5 * vol_ma20[i] if not np.isnan(vol_ma20[i]) else False
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean reversion)
        chop_trend = chop[i] < 45.0   # Trending (breakout)
        
        # === RSI EXTREMES (7-period for sensitivity) ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        rsi_extreme_oversold = rsi_7[i] < 20.0
        rsi_extreme_overbought = rsi_7[i] > 80.0
        
        # === CAMARILLA LEVEL TESTS ===
        at_s3 = low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]
        at_s4 = low[i] <= s4_aligned[i] and close[i] > s4_aligned[i]
        at_r3 = high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]
        at_r4 = high[i] >= r4_aligned[i] and close[i] < r4_aligned[i]
        
        below_s3 = close[i] < s3_aligned[i]
        above_r3 = close[i] > r3_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at Camarilla S3/R3 with RSI extreme
        if chop_range and in_session:
            # Long: at S3 + RSI oversold + HTF bull or neutral
            if at_s3 and rsi_extreme_oversold and (htf_bull or htf_neutral):
                if vol_spike:
                    desired_signal = SIZE_STRONG
                elif rsi_7[i] < rsi_7[i-1] if i > 0 else False:
                    desired_signal = SIZE_BASE
            # Short: at R3 + RSI overbought + HTF bear or neutral
            elif at_r3 and rsi_extreme_overbought and (htf_bear or htf_neutral):
                if vol_spike:
                    desired_signal = -SIZE_STRONG
                elif rsi_7[i] > rsi_7[i-1] if i > 0 else False:
                    desired_signal = -SIZE_BASE
        
        # TREND REGIME: Breakout at Camarilla S4/R4 with HTF confirmation
        elif chop_trend and in_session:
            # Long breakout: above R4 + HTF bull + volume spike
            if above_r4 and htf_bull and vol_spike and rsi_7[i] > 50.0:
                desired_signal = SIZE_STRONG
            # Short breakout: below S4 + HTF bear + volume spike
            elif below_s3 and htf_bear and vol_spike and rsi_7[i] < 50.0:
                desired_signal = -SIZE_STRONG
            # Pullback entry in trend
            elif htf_bull and rsi_oversold and close[i] > hma_4h_aligned[i]:
                if vol_spike:
                    desired_signal = SIZE_BASE
            elif htf_bear and rsi_overbought and close[i] < hma_4h_aligned[i]:
                if vol_spike:
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