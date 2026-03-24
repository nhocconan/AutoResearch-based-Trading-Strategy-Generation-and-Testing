#!/usr/bin/env python3
"""
Experiment #649: 15m Primary + 1h/1d HTF — Daily CPR Breakout + HMA Trend + RSI Momentum

Hypothesis: 15m timeframe with Daily CPR (Central Pivot Range) breakout strategy.
Narrow CPR indicates consolidation, breakout in HTF trend direction captures expansion.
Key innovations:
1. Daily CPR from 1d HTF - BC/TC levels act as support/resistance
2. Narrow CPR filter - only trade when (TC-BC)/Close < 1.5% (consolidation)
3. 1h HMA(21) trend filter - long only above, short only below
4. RSI(7) momentum - entry on pullback (RSI 35-55 for long, 45-65 for short)
5. Session filter - only 00-12 UTC (London/NY overlap, highest liquidity)
6. ATR(14) trailing stop - 2.5x for risk management

Why this works for 15m:
- CPR breakout = fewer false signals than simple EMA cross
- Narrow CPR filter = only trade when volatility compressed (expansion imminent)
- 1h HMA = HTF trend alignment reduces whipsaw
- Session filter = avoid low-liquidity Asian session chop
- Target: 50-80 trades/year, Sharpe > 0.40

Position sizing: 0.15-0.20 (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_breakout_hma_rsi_1h1d_session_v1"
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Pad to match length
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
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

def calculate_cpr_from_ohlc(open_p, high, low, close):
    """
    Calculate Central Pivot Range from daily OHLC
    Pivot = (H + L + C) / 3
    BC = (H + L) / 2
    TC = (Pivot - BC) + Pivot = 2*Pivot - BC
    """
    n = len(close)
    pivot = np.full(n, np.nan)
    bc = np.full(n, np.nan)
    tc = np.full(n, np.nan)
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        bc[i] = (high[i] + low[i]) / 2.0
        tc[i] = 2.0 * pivot[i] - bc[i]
    
    return pivot, bc, tc

def calculate_cpr_width_ratio(tc, bc, close):
    """CPR width as percentage of price - narrow = consolidation"""
    n = len(close)
    width_ratio = np.full(n, np.nan)
    for i in range(n):
        if close[i] > 1e-10 and not np.isnan(tc[i]) and not np.isnan(bc[i]):
            width_ratio[i] = abs(tc[i] - bc[i]) / close[i]
    return width_ratio

def get_hour_from_open_time(open_time_col):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_col // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1d CPR levels
    pivot_1d, bc_1d, tc_1d = calculate_cpr_from_ohlc(
        df_1d['open'].values,
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Calculate CPR width ratio for narrow filter
    cpr_width_1d = calculate_cpr_width_ratio(tc_1d, bc_1d, df_1d['close'].values)
    
    # Align 1d CPR to 15m (shifted by 1 day = use yesterday's CPR for today)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    cpr_width_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d)
    
    # Calculate and align 1h HMA for trend filter
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bc_aligned[i]) or np.isnan(tc_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(cpr_width_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        in_session = hours[i] >= 0 and hours[i] <= 12
        
        # === NARROW CPR FILTER (< 1.5% width = consolidation) ===
        narrow_cpr = cpr_width_aligned[i] < 0.015
        
        # === HTF TREND (1h HMA) ===
        htf_bull = close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_1h_aligned[i]
        
        # === CPR BREAKOUT SIGNALS ===
        # Long: price breaks above TC (resistance becomes support)
        breakout_long = close[i] > tc_aligned[i]
        # Short: price breaks below BC (support becomes resistance)
        breakout_short = close[i] < bc_aligned[i]
        
        # === RSI MOMENTUM (pullback entry, not extreme) ===
        # For long: RSI 35-55 (oversold bounce, not overbought)
        rsi_long_ok = 35.0 <= rsi_7[i] <= 55.0
        # For short: RSI 45-65 (overbought pullback, not oversold)
        rsi_short_ok = 45.0 <= rsi_7[i] <= 65.0
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: session + narrow CPR + HTF bull + TC breakout + RSI ok
        long_confluence = sum([in_session, narrow_cpr, htf_bull, breakout_long, rsi_long_ok])
        if long_confluence >= 4:
            if htf_bull and breakout_long:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: session + narrow CPR + HTF bear + BC breakout + RSI ok
        short_confluence = sum([in_session, narrow_cpr, htf_bear, breakout_short, rsi_short_ok])
        if short_confluence >= 4:
            if htf_bear and breakout_short:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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