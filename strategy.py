#!/usr/bin/env python3
"""
Experiment #805: 15m Primary + 4h/1d HTF — Daily Pivot + RSI Mean Reversion

Hypothesis: 15m timeframe with 4h/1d HTF filters can capture intraday mean reversion
opportunities while avoiding whipsaws. Key insight from failed 15m experiments:
- Too strict = 0 trades (experiments #796, #797, #799, #801 all Sharpe=0.000)
- Too loose = fee death (>300 trades/year)

Solution: 3+ confluence filters for SELECTIVE entries (target 40-100 trades/year):
1. Daily pivot levels (R1, S1, PP) from 1d HTF — defines "value area"
2. 4h HMA(21) for trend bias — only trade with HTF trend
3. 15m RSI(7) for quick mean reversion — faster than RSI(14)
4. Session filter (00-12 UTC) — London/NY overlap for crypto liquidity
5. Choppiness Index — avoid choppy markets (CHOP > 61.8 = skip)

Position sizing: 0.15-0.20 (smaller for 15m frequency)
Stoploss: ATR(14) 2.0x trailing
Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pivot_rsi_session_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending (46.4-61.8 range)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chopp = np.zeros(n)
    chopp[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if highest_high - lowest_low > 1e-10:
            chopp[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chopp

def calculate_daily_pivots(open_prices, high_prices, low_prices, close_prices):
    """Calculate daily pivot points (PP, R1, S1, R2, S2)"""
    n = len(close_prices)
    pp = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    r2 = np.zeros(n)
    s2 = np.zeros(n)
    
    # Track current day's OHLC
    day_open = open_prices[0]
    day_high = high_prices[0]
    day_low = low_prices[0]
    day_close = close_prices[0]
    day_idx = 0
    
    for i in range(n):
        # Update day's OHLC
        if i == 0 or open_prices[i] != day_open:
            # New day - calculate pivots from previous day
            if day_idx > 0:
                pp[day_idx:i] = (day_high + day_low + day_close) / 3.0
                r1[day_idx:i] = 2.0 * pp[day_idx] - day_low
                s1[day_idx:i] = 2.0 * pp[day_idx] - day_high
                r2[day_idx:i] = pp[day_idx] + (day_high - day_low)
                s2[day_idx:i] = pp[day_idx] - (day_high - day_low)
            
            # Reset for new day
            day_open = open_prices[i]
            day_high = high_prices[i]
            day_low = low_prices[i]
            day_close = close_prices[i]
            day_idx = i
        else:
            day_high = max(day_high, high_prices[i])
            day_low = min(day_low, low_prices[i])
            day_close = close_prices[i]
    
    # Fill remaining
    if day_idx < n:
        pp[day_idx:n] = (day_high + day_low + day_close) / 3.0
        r1[day_idx:n] = 2.0 * pp[day_idx] - day_low
        s1[day_idx:n] = 2.0 * pp[day_idx] - day_high
        r2[day_idx:n] = pp[day_idx] + (day_high - day_low)
        s2[day_idx:n] = pp[day_idx] - (day_high - day_low)
    
    return pp, r1, s1, r2, s2

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_p = prices["open"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    chopp_14 = calculate_choppiness(high, low, close, period=14)
    
    # Calculate daily pivots from 15m data
    pp_15m, r1_15m, s1_15m, r2_15m, s2_15m = calculate_daily_pivots(open_p, high, low, close)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chopp_14[i]):
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
        
        # === SESSION FILTER (00-12 UTC) ===
        # 15m bars: 96 per day. Hours 0-12 = bars 0-48 of each day
        bar_in_day = i % 96
        in_session = bar_in_day < 48  # First 12 hours (00:00-12:00 UTC)
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === CHOPPINESS FILTER ===
        # CHOP > 61.8 = choppy/range (skip), CHOP < 38.2 = trending
        is_choppy = chopp_14[i] > 55.0  # Slightly relaxed for more trades
        
        # === PIVOT LEVELS ===
        pp = pp_15m[i]
        r1 = r1_15m[i]
        s1 = s1_15m[i]
        
        # Price position relative to pivots
        near_s1 = (low[i] <= s1 * 1.005) and (close[i] > s1)  # Touched S1, closed above
        near_r1 = (high[i] >= r1 * 0.995) and (close[i] < r1)  # Touched R1, closed below
        above_pp = close[i] > pp
        below_pp = close[i] < pp
        
        # === RSI CONDITIONS (FAST RSI(7)) ===
        rsi_oversold = rsi_7[i] < 35.0  # Faster than RSI(14) 30
        rsi_overbought = rsi_7[i] > 65.0  # Faster than RSI(14) 70
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + (near S1 OR RSI oversold) + session + not choppy
        if htf_strong_bull or htf_4h_bull:
            confluence_count = 0
            if near_s1:
                confluence_count += 1
            if rsi_oversold:
                confluence_count += 1
            if in_session:
                confluence_count += 1
            if below_pp:  # In value area
                confluence_count += 1
            
            if confluence_count >= 2 and not is_choppy:
                if rsi_extreme_oversold or near_s1:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + (near R1 OR RSI overbought) + session + not choppy
        elif htf_strong_bear or htf_4h_bear:
            confluence_count = 0
            if near_r1:
                confluence_count += 1
            if rsi_overbought:
                confluence_count += 1
            if in_session:
                confluence_count += 1
            if above_pp:  # In value area
                confluence_count += 1
            
            if confluence_count >= 2 and not is_choppy:
                if rsi_extreme_overbought or near_r1:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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