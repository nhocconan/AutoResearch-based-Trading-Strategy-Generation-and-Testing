#!/usr/bin/env python3
"""
EXPERIMENT #030 - MTF KAMA+RSI+VOLUME 30m+4h v1
==================================================================================================
Hypothesis: Simplify the MTF approach that worked in #022 (30m+4h) but use KAMA (adaptive MA)
instead of HMA for better volatility adaptation. Add volume confirmation to filter weak moves.
Key differences from current best:
- KAMA (Kaufman Adaptive) instead of HMA - adapts to market efficiency
- Volume filter - only trade when volume confirms the move
- Simpler entry logic - reduce churn and fees
- 30m base timeframe (proven in #022 with Sharpe=1.153)

Why this should work:
- KAMA adjusts smoothing based on market noise (ER ratio)
- 30m has fewer false signals than 15m (less noise)
- Volume confirmation filters out low-conviction moves
- 4h trend filter prevents counter-trend trades
- Simpler logic = fewer signal changes = lower fees
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_volume_30m_4h_v1"
timeframe = "30m"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    """
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
    
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    kama_30m = calculate_kama(close, period=10, fast=2, slow=30)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        v_4h = df_4h['volume'].values
        
        # 4h KAMA for trend direction
        kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
        rsi_4h = calculate_rsi(c_4h, period=14)
        vol_sma_4h = calculate_volume_sma(v_4h, period=20)
        
        # Align 4h indicators to 30m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
        vol_sma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_4h)
    except Exception:
        kama_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n)
        vol_sma_4h_aligned = np.zeros(n)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Volume filter - must be above average
    VOL_MULT = 1.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Minimum bars for valid signals
    first_valid = max(200, 14 * 2, 30, 40)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i - 1]
            continue
        
        # Get aligned MTF values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        vol_4h_avg = vol_sma_4h_aligned[i] if i < len(vol_sma_4h_aligned) else 0
        
        # Volume filter on 4h (avoid low volume periods)
        if vol_4h_avg > 0:
            vol_ratio = volume[i] / vol_4h_avg if vol_4h_avg > 0 else 0
        else:
            vol_ratio = 1.0
        
        # Determine 4h trend (price vs KAMA)
        trend_4h = 0
        if kama_4h_val > 0:
            if close[i] > kama_4h_val:
                trend_4h = 1
            elif close[i] < kama_4h_val:
                trend_4h = -1
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
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
        
        # Entry logic: 4h trend + 30m KAMA + RSI pullback + Volume
        price = close[i]
        
        # Volume confirmation (must be at least average)
        volume_ok = vol_ratio >= VOL_MULT
        
        # 4h trend filter
        trend_bullish = (trend_4h == 1 and rsi_4h_val > 50)
        trend_bearish = (trend_4h == -1 and rsi_4h_val < 50)
        
        # 30m KAMA trend (price vs KAMA)
        kama_trend_bullish = (price > kama_30m[i]) if kama_30m[i] > 0 else False
        kama_trend_bearish = (price < kama_30m[i]) if kama_30m[i] > 0 else False
        
        if trend_bullish and kama_trend_bullish and volume_ok:
            # Long entry: RSI pullback (not overbought)
            if RSI_LONG_MIN <= rsi_30m[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_bearish and kama_trend_bearish and volume_ok:
            # Short entry: RSI pullback (not oversold)
            if RSI_SHORT_MIN <= rsi_30m[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals