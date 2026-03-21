#!/usr/bin/env python3
"""
EXPERIMENT #046 - DEMA MACD ADX Momentum with 4h Trend Filter (30m Primary)
==================================================================================================
Hypothesis: Current best (hma_rsi_pullback_daily_trend_4h_v1) uses 4h primary + 1d filter with Sharpe=0.537.
This strategy uses 30m primary + 4h HTF for MORE trade opportunities while maintaining trend quality.
DEMA responds faster than HMA/KAMA to trend changes. MACD histogram provides momentum confirmation.
ADX filters out choppy markets (only trade when ADX > 20 = real trend).

Key innovations:
1. 30m PRIMARY + 4h HTF: More trades than 1h/4h primary strategies
2. DEMA for fast trend: Double EMA reduces lag vs single EMA/HMA
3. MACD histogram momentum: Confirms entry timing with momentum surge
4. ADX trend strength filter: Avoid trading in choppy/ranging markets (ADX < 20)
5. RSI momentum zone: RSI 45-65 for entries (not extreme, just momentum)
6. 2.0*ATR stoploss: Wider than #045's 1.5*ATR to reduce premature stops
7. Position sizing: Discrete levels 0.0, ±0.20, ±0.30 (max 0.35)

Why this should beat hma_rsi_pullback_daily_trend_4h_v1 (Sharpe=0.537):
- 30m captures more intraday moves than 4h primary
- DEMA + MACD combo more responsive than HMA alone
- ADX filter avoids whipsaw trades in ranging markets
- More trades = better statistical significance for Sharpe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_macd_adx_momentum_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_dema(close, period=21):
    """
    Double Exponential Moving Average
    DEMA = 2*EMA1 - EMA2(EMA1)
    Reduces lag compared to standard EMA
    """
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    dema = 2 * ema1 - ema2
    return dema.values


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_s.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line.values, macd_signal.values, macd_hist.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 30m INDICATORS (PRIMARY TIMEFRAME) ==========
    dema_30m_fast = calculate_dema(close, period=8)
    dema_30m_slow = calculate_dema(close, period=21)
    macd_line_30m, macd_signal_30m, macd_hist_30m = calculate_macd(close, fast=12, slow=26, signal=9)
    rsi_30m = calculate_rsi(close, period=14)
    adx_30m = calculate_adx(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        dema_4h = calculate_dema(close_4h, period=21)
        adx_4h = calculate_adx(high_4h, low_4h, close_4h, period=14)
        atr_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        dema_4h_aligned = align_htf_to_ltf(prices, df_4h, dema_4h)
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
        atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
        
    except Exception:
        dema_4h_aligned = np.zeros(n)
        adx_4h_aligned = np.zeros(n)
        atr_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE LEVELS ONLY
    SIZE_BASE = 0.20   # Base position (20% of capital)
    SIZE_HIGH = 0.30   # High conviction (30% of capital)
    SIZE_MAX = 0.35    # Maximum position (35% of capital)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    # RSI momentum zones
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    
    # ADX threshold for trend strength
    ADX_MIN_30M = 20
    ADX_MIN_4H = 15
    
    first_valid = 200
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0 or np.isnan(rsi_30m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        rsi_val = rsi_30m[i]
        adx_val = adx_30m[i]
        macd_hist_val = macd_hist_30m[i]
        dema_fast_val = dema_30m_fast[i]
        dema_slow_val = dema_30m_slow[i]
        
        # 4h trend filters (MASTER FILTER)
        dema_4h_val = dema_4h_aligned[i]
        adx_4h_val = adx_4h_aligned[i]
        
        # Determine 4h trend direction
        trend_4h = 0
        if dema_4h_val > 0:
            if price > dema_4h_val:
                trend_4h = 1
            elif price < dema_4h_val:
                trend_4h = -1
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Trail stop at 1R profit after TP triggered
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE  # Reduce from 0.30 to 0.20
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Trail stop at 1R profit after TP triggered
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = False
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE  # Reduce from -0.30 to -0.20
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== ENTRY LOGIC - MOMENTUM IN TREND DIRECTION ==========
        # 30m trend: DEMA fast > DEMA slow for long
        trend_30m_long = dema_fast_val > dema_slow_val and dema_slow_val > 0
        trend_30m_short = dema_fast_val < dema_slow_val and dema_slow_val > 0
        
        # LONG: 4h trend up + 30m trend up + MACD momentum + RSI in zone + ADX confirms trend
        long_condition = (
            trend_4h == 1 and
            adx_4h_val > ADX_MIN_4H and
            trend_30m_long and
            macd_hist_val > 0 and
            rsi_val > RSI_LONG_MIN and
            rsi_val < RSI_LONG_MAX and
            adx_val > ADX_MIN_30M
        )
        
        # SHORT: 4h trend down + 30m trend down + MACD momentum + RSI in zone + ADX confirms trend
        short_condition = (
            trend_4h == -1 and
            adx_4h_val > ADX_MIN_4H and
            trend_30m_short and
            macd_hist_val < 0 and
            rsi_val > RSI_SHORT_MIN and
            rsi_val < RSI_SHORT_MAX and
            adx_val > ADX_MIN_30M
        )
        
        # Determine position size based on conviction
        # High conviction: strong ADX on both timeframes
        high_conviction_long = long_condition and adx_4h_val > 25 and adx_val > 25
        high_conviction_short = short_condition and adx_4h_val > 25 and adx_val > 25
        
        if long_condition:
            size = SIZE_HIGH if high_conviction_long else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_condition:
            size = SIZE_HIGH if high_conviction_short else SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals