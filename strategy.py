#!/usr/bin/env python3
"""
Experiment #1421: 15m Primary + 1h/4h/1d HTF — Regime-Adaptive Pivot Breakout

Hypothesis: 15m has failed 2x before (#1409, #1417) because trend-following gets 
whipsawed. This strategy uses REGIME-ADAPTIVE logic:
1. 1d CPR (Central Pivot Range) for key S/R levels — proven institutional levels
2. 4h HMA(21) for major trend bias
3. 4h Choppiness Index for regime detection (trend vs range)
4. 15m RSI(7) for entry timing (extremes only: <25 or >75)
5. Session filter: 00-12 UTC only (London/NY overlap = high volume)
6. ATR(14) trailing stoploss

Why this should work where pure HMA/RSI failed:
- CPR levels are REAL support/resistance (not arbitrary indicators)
- Choppiness filter avoids trading in whipsaw conditions
- Session filter cuts 50% of trades (only high-volume hours)
- RSI(7) extremes are RARE on 15m = fewer trades, higher quality
- Size 0.15-0.20 (smaller than 4h strategies due to higher frequency)

Entry logic (VERY SELECTIVE — target 50-80 trades/year):
- LONG: 4h_HMA bullish + CHOP<50 (trending) + 15m_RSI<25 + price>daily_BC
- SHORT: 4h_HMA bearish + CHOP<50 (trending) + 15m_RSI>75 + price<daily_TC
- RANGE: CHOP>61.8 + fade at CPR extremes (mean-reversion mode)

Timeframe: 15m (FIRST 15m experiment — high priority)
Size: 0.15-0.20 discrete (smaller for higher frequency)
Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_cpr_pivot_rsi_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
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
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_cpr_levels(df_daily):
    """
    Calculate Daily CPR (Central Pivot Range) levels
    Returns: pivot, BC (bottom central), TC (top central) arrays aligned to daily bars
    """
    n = len(df_daily)
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        prev_high = df_daily['high'].iloc[i-1]
        prev_low = df_daily['low'].iloc[i-1]
        prev_close = df_daily['close'].iloc[i-1]
        
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        bc[i] = (prev_high + prev_low) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
    
    return pivot, bc, tc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d, bc_1d, tc_1d = calculate_cpr_levels(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_4h_aligned[i]) or np.isnan(bc_aligned[i]) or np.isnan(tc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop = chop_4h_aligned[i]
        is_trending = chop < 50.0  # Below 50 = trending
        is_ranging = chop > 61.8  # Above 61.8 = ranging/choppy
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CPR LEVELS (daily support/resistance) ===
        bc = bc_aligned[i]
        tc = tc_aligned[i]
        price_above_bc = close[i] > bc
        price_below_tc = close[i] < tc
        
        # === RSI EXTREMES (15m timing) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 25
        rsi_overbought = rsi > 75
        
        # === ENTRY LOGIC (VERY SELECTIVE) ===
        desired_signal = 0.0
        
        # TREND MODE (CHOP < 50): Trade breakouts in trend direction
        if is_trending and in_session:
            # LONG: 4h bullish + RSI oversold + price above daily BC
            if price_above_4h and rsi_oversold and price_above_bc:
                desired_signal = SIZE_STRONG
            
            # SHORT: 4h bearish + RSI overbought + price below daily TC
            elif price_below_4h and rsi_overbought and price_below_tc:
                desired_signal = -SIZE_STRONG
        
        # RANGE MODE (CHOP > 61.8): Mean-reversion at CPR extremes
        elif is_ranging and in_session:
            # LONG: Price near BC + RSI oversold (fade the low)
            if rsi_oversold and (close[i] - bc) / bc < 0.01:  # Within 1% of BC
                desired_signal = SIZE_BASE
            
            # SHORT: Price near TC + RSI overbought (fade the high)
            elif rsi_overbought and (tc - close[i]) / tc < 0.01:  # Within 1% of TC
                desired_signal = -SIZE_BASE
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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