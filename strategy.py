#!/usr/bin/env python3
"""
EXPERIMENT #117 - MTF Supertrend+RSI+Chandelier+VolAdj (15m entries, 4h trend, proper HTF)
==================================================================================================
Hypothesis: Recent failures (#105-#116) all had massive drawdowns due to:
1. Improper MTF alignment (manual resampling vs mtf_data helper)
2. Too many signal changes (churning fees)
3. Position sizing not volatility-adjusted

This strategy uses:
- 15m for entries (proven in #031, #034, #035 with Sharpe > 7.5)
- 4h trend filter via mtf_data helper (proper HTF alignment)
- Supertrend for trend direction
- RSI for pullback entries
- Chandelier exit (3*ATR) for stops
- Volatility-adjusted position sizing (reduce size in high vol regimes)
- Discrete signal levels (0, ±0.20, ±0.35) to minimize churning

Why this should beat #116 (Sharpe=-2.542):
- Proper mtf_data usage (no manual resampling bugs)
- Simpler logic (fewer failure points)
- Volatility-adjusted sizing (smaller positions in high vol = lower DD)
- Based on proven #040 framework (Sharpe=5.4)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_chandelier_voladj_15m_4h_proper_v1"
timeframe = "15m"
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


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_bbw(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        
        if mean > 0:
            bbw[i] = (2 * std_mult * std) / mean
        else:
            bbw[i] = 0
    
    return bbw


def calculate_vol_regime(atr, close, lookback=50):
    """
    Calculate volatility regime (0=low, 1=medium, 2=high)
    Based on ATR percentile over lookback period
    """
    n = len(close)
    regime = np.zeros(n)
    
    for i in range(lookback, n):
        atr_window = atr[i - lookback + 1:i + 1]
        atr_pct = atr[i] / close[i] if close[i] > 0 else 0
        
        # Calculate percentile of current ATR% in recent history
        atr_pct_window = atr_window / close[i - lookback + 1:i + 1]
        percentile = np.sum(atr_pct_window < atr_pct) / lookback
        
        if percentile < 0.33:
            regime[i] = 0  # Low vol
        elif percentile < 0.67:
            regime[i] = 1  # Medium vol
        else:
            regime[i] = 2  # High vol
    
    return regime


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ===== 15m indicators (entry timeframe) =====
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    bbw_15m = calculate_bbw(close, period=20, std_mult=2.0)
    vol_regime = calculate_vol_regime(atr_15m, close, lookback=50)
    
    # ===== 4h indicators (trend filter) using mtf_data helper =====
    # CRITICAL: Use mtf_data helper for proper HTF alignment
    df_4h = get_htf_data(prices, '4h')
    
    if df_4h is None or len(df_4h) < 50:
        # Fallback if 4h data not available
        st_dir_4h_aligned = np.ones(n)
        rsi_4h_aligned = np.zeros(n)
    else:
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        _, st_dir_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        rsi_4h = calculate_rsi(close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        st_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, st_dir_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # ===== Position sizing parameters =====
    # DISCRETE levels to avoid churning (CRITICAL for fee control)
    SIZE_LOW_VOL = 0.35   # Low volatility = larger position
    SIZE_MED_VOL = 0.25   # Medium volatility
    SIZE_HIGH_VOL = 0.15  # High volatility = smaller position (reduce risk)
    
    # Stoploss and take profit
    ATR_STOP_MULT = 3.0   # Chandelier exit (3*ATR)
    TP_MULT = 2.0         # Take profit at 2R
    
    # Entry filters
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    first_valid = max(100, 50)  # Warmup period
    
    # ===== Generate signals =====
    signals = np.zeros(n)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    tp_triggered = np.zeros(n, dtype=bool)
    
    for i in range(first_valid, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        st_15m = st_dir_15m[i]
        st_4h = st_dir_4h_aligned[i]
        rsi_4h = rsi_4h_aligned[i]
        regime = vol_regime[i]
        
        # Determine position size based on volatility regime
        if regime == 0:
            base_size = SIZE_LOW_VOL
        elif regime == 1:
            base_size = SIZE_MED_VOL
        else:
            base_size = SIZE_HIGH_VOL
        
        # ===== Check existing position for exits =====
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = prev_low if prev_low > 0 else price
            else:
                current_high = prev_high if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Chandelier exit (stoploss)
            if prev_side == 1:
                stoploss_price = current_high - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = base_size * 0.5
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP triggered
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
                stoploss_price = current_low + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = False
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -base_size * 0.5
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = True
                    continue
                
                # Trail stop after TP triggered
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
        
        # ===== Entry logic =====
        # 4h Supertrend must agree with 15m Supertrend (MTF confirmation)
        if st_4h == 0 or st_15m == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Long entry: 4h and 15m Supertrend bullish + RSI pullback
        if st_4h == 1 and st_15m == 1:
            if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                signals[i] = base_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                continue
        
        # Short entry: 4h and 15m Supertrend bearish + RSI pullback
        elif st_4h == -1 and st_15m == -1:
            if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                signals[i] = -base_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = False
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                continue
        
        # No entry signal
        signals[i] = 0.0
        position_side[i] = 0
    
    return signals