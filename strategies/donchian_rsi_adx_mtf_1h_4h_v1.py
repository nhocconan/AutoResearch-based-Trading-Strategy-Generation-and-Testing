#!/usr/bin/env python3
"""
EXPERIMENT #002 - Donchian Trend + RSI Pullback with ADX Filter (1h Primary, 4h HTF)
==================================================================================================
Hypothesis: Previous KAMA/Supertrend ensemble approaches failed due to over-complexity and 
laggy adaptive indicators. This version uses simpler, proven components:

1. DONCHIAN CHANNEL (20-period): Clean trend identification based on price breakouts
2. RSI (14) PULLBACK: Enter on RSI dips in uptrend (40-50) and rallies in downtrend (50-60)
3. ADX (14) FILTER: Only trade when ADX > 25 (trending market, not ranging)
4. 4h HTF TREND: Master filter - only trade 1h signals that align with 4h Donchian trend

Why this should work better than #001 (KAMA+HMA+RSI+Z-score, Sharpe=-0.042):
- Donchian is cleaner than KAMA for trend identification (less lag, clear breakouts)
- RSI pullback entries are proven (buy dips in uptrend, not breakouts)
- ADX filter avoids whipsaws in ranging markets (major failure mode in #001)
- Simpler logic = fewer parameters to overfit
- 1h primary timeframe = fewer false signals than 15m, more than 4h

Risk Management:
- Position size: 0.20-0.35 (discrete levels)
- Stoploss: 2.0 ATR trailing stop (signal→0)
- Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_rsi_adx_mtf_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


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


def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over N periods
    Returns: upper_band, lower_band, middle_band
    """
    n = len(close) if 'close' in dir() else len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength
    ADX > 25 = trending market, ADX < 25 = ranging market
    """
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth +DM, -DM, and TR using Wilder's method
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 0
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 1h INDICATORS (ENTRY TIMING) ==========
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    donchian_upper_1h, donchian_mid_1h, donchian_lower_1h = calculate_donchian(high, low, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h Donchian for master trend
        donchian_upper_4h, donchian_mid_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
        
        # Align to 1h timeframe (auto shift for completed bars)
        donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
        
    except Exception:
        donchian_mid_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.20    # Standard position
    SIZE_HIGH = 0.35    # High conviction (all filters agree)
    
    # Filter thresholds
    ADX_TREND_THRESHOLD = 25.0  # ADX > 25 = trending market
    RSI_PULLBACK_LONG_LOW = 40.0
    RSI_PULLBACK_LONG_HIGH = 50.0
    RSI_PULLBACK_SHORT_LOW = 50.0
    RSI_PULLBACK_SHORT_HIGH = 60.0
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 40)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(rsi_1h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1h[i]
        rsi_val = rsi_1h[i]
        adx_val = adx_1h[i]
        
        # Donchian trend on 1h
        donchian_trend_1h = 0
        if price > donchian_mid_1h[i]:
            donchian_trend_1h = 1
        elif price < donchian_mid_1h[i]:
            donchian_trend_1h = -1
        
        # 4h trend filter (master)
        donchian_4h_val = donchian_mid_4h_aligned[i]
        donchian_trend_4h = 0
        if donchian_4h_val > 0 and price > donchian_4h_val:
            donchian_trend_4h = 1
        elif donchian_4h_val > 0 and price < donchian_4h_val:
            donchian_trend_4h = -1
        
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_BASE  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_BASE  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop at 1R profit
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
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== NEW ENTRY LOGIC ==========
        # Must have trending market (ADX > 25)
        if adx_val < ADX_TREND_THRESHOLD:
            signals[i] = 0.0
            continue
        
        # 4h trend must agree with entry direction (cross-asset filter)
        # Long setup: 4h trend up + 1h trend up + RSI pullback
        long_setup = (
            donchian_trend_4h == 1 and
            donchian_trend_1h == 1 and
            rsi_val >= RSI_PULLBACK_LONG_LOW and
            rsi_val <= RSI_PULLBACK_LONG_HIGH
        )
        
        # Short setup: 4h trend down + 1h trend down + RSI pullback
        short_setup = (
            donchian_trend_4h == -1 and
            donchian_trend_1h == -1 and
            rsi_val >= RSI_PULLBACK_SHORT_LOW and
            rsi_val <= RSI_PULLBACK_SHORT_HIGH
        )
        
        # Determine position size
        # High conviction: strong ADX (>35) + clear trend alignment
        high_conviction = adx_val > 35.0 and donchian_trend_4h == donchian_trend_1h
        
        if long_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        elif short_setup:
            size = SIZE_HIGH if high_conviction else SIZE_BASE
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = False
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
    
    return signals