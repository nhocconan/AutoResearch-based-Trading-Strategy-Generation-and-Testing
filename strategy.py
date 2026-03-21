#!/usr/bin/env python3
"""
EXPERIMENT #118 - MTF Supertrend+RSI+ADX+ATR Trail (15m entries, 1h trend, optimized)
==================================================================================================
Hypothesis: #117 barely passed (Sharpe=0.078). Issues identified:
1. 4h trend filter too slow for crypto volatility
2. 3*ATR stop too wide (causes large drawdowns before exit)
3. Too many signal state variables causing bugs

This strategy uses:
- 15m for entries (proven in #031, #034, #035 with Sharpe > 7.5)
- 1h trend filter (faster response than 4h, better for crypto)
- ADX > 25 filter (avoid choppy/ranging markets - CRITICAL for fee reduction)
- Supertrend for trend direction
- RSI for pullback entries (buy dips in uptrend)
- 2*ATR Chandelier exit (tighter stops = lower drawdown)
- ATR% volatility regime for position sizing
- Discrete signal levels (0, ±0.25, ±0.35) to minimize churning

Why this should beat #117 (Sharpe=0.078) and approach #114's best (Sharpe=3.653):
- 1h trend filter = faster entries, less lag than 4h
- ADX filter = fewer trades in choppy markets (saves fees)
- 2*ATR stop = tighter risk control, lower drawdown
- Simpler state tracking = fewer bugs
- Based on proven 15m entry framework from #031/#034/#035
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_adx_atrtrail_15m_1h_optimized_v1"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    avg_plus_dm = np.zeros(n)
    avg_minus_dm = np.zeros(n)
    avg_tr = np.zeros(n)
    
    avg_plus_dm[period - 1] = np.mean(plus_dm[1:period])
    avg_minus_dm[period - 1] = np.mean(minus_dm[1:period])
    avg_tr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        avg_plus_dm[i] = (avg_plus_dm[i - 1] * (period - 1) + plus_dm[i]) / period
        avg_minus_dm[i] = (avg_minus_dm[i - 1] * (period - 1) + minus_dm[i]) / period
        avg_tr[i] = (avg_tr[i - 1] * (period - 1) + tr[i]) / period
    
    for i in range(period, n):
        if avg_tr[i] > 0:
            plus_di[i] = 100 * avg_plus_dm[i] / avg_tr[i]
            minus_di[i] = 100 * avg_minus_dm[i] / avg_tr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.zeros(n)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_atr_pct(atr, close):
    """Calculate ATR as percentage of price"""
    n = len(close)
    atr_pct = np.zeros(n)
    for i in range(n):
        if close[i] > 0:
            atr_pct[i] = atr[i] / close[i] * 100
    return atr_pct


def calculate_vol_regime(atr_pct, lookback=50):
    """
    Calculate volatility regime based on ATR% percentile
    0=low, 1=medium, 2=high
    """
    n = len(atr_pct)
    regime = np.ones(n)  # Default to medium
    
    for i in range(lookback, n):
        window = atr_pct[i - lookback + 1:i + 1]
        current = atr_pct[i]
        
        percentile = np.sum(window < current) / lookback
        
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
    atr_pct_15m = calculate_atr_pct(atr_15m, close)
    rsi_15m = calculate_rsi(close, period=14)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_15m = calculate_adx(high, low, close, period=14)
    vol_regime = calculate_vol_regime(atr_pct_15m, lookback=50)
    
    # ===== 1h indicators (trend filter) using mtf_data helper =====
    df_1h = get_htf_data(prices, '1h')
    
    if df_1h is None or len(df_1h) < 50:
        st_dir_1h_aligned = np.ones(n)
        adx_1h_aligned = np.zeros(n)
    else:
        close_1h = df_1h['close'].values
        high_1h = df_1h['high'].values
        low_1h = df_1h['low'].values
        
        _, st_dir_1h = calculate_supertrend(high_1h, low_1h, close_1h, period=10, multiplier=3.0)
        adx_1h = calculate_adx(high_1h, low_1h, close_1h, period=14)
        
        st_dir_1h_aligned = align_htf_to_ltf(prices, df_1h, st_dir_1h)
        adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # ===== Position sizing parameters =====
    SIZE_LOW_VOL = 0.35
    SIZE_MED_VOL = 0.25
    SIZE_HIGH_VOL = 0.15
    
    # Stoploss and take profit (tighter than #117)
    ATR_STOP_MULT = 2.0   # 2*ATR stop (tighter = lower drawdown)
    TP_MULT = 2.0         # Take profit at 2R
    
    # Entry filters
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    ADX_MIN = 25          # Only trade when trend is strong
    
    first_valid = max(100, 50)
    
    # ===== Generate signals =====
    signals = np.zeros(n)
    
    # Simplified position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    tp_triggered = False
    
    for i in range(first_valid, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi = rsi_15m[i]
        st_15m = st_dir_15m[i]
        st_1h = st_dir_1h_aligned[i]
        adx_1h = adx_1h_aligned[i]
        regime = vol_regime[i]
        
        # Determine position size based on volatility regime
        if regime == 0:
            base_size = SIZE_LOW_VOL
        elif regime == 1:
            base_size = SIZE_MED_VOL
        else:
            base_size = SIZE_HIGH_VOL
        
        # ===== Check existing position for exits =====
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                stoploss_price = highest_since_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    # Stoploss triggered
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    continue
                
                # Take profit at 2R
                tp_price = entry_price + TP_MULT * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = base_size * 0.5
                    tp_triggered = True
                    continue
                
                # Trail stop after TP
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        continue
                
                signals[i] = base_size if not tp_triggered else base_size * 0.5
                
            elif position_side == -1:
                lowest_since_entry = min(lowest_since_entry, price)
                stoploss_price = lowest_since_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    lowest_since_entry = 0.0
                    continue
                
                tp_price = entry_price - TP_MULT * ATR_STOP_MULT * entry_atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -base_size * 0.5
                    tp_triggered = True
                    continue
                
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        lowest_since_entry = 0.0
                        continue
                
                signals[i] = -base_size if not tp_triggered else -base_size * 0.5
            
            continue
        
        # ===== Entry logic =====
        # ADX filter: only trade when trend is strong (avoid choppy markets)
        if adx_1h < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # Supertrend must agree on both timeframes
        if st_1h == 0 or st_15m == 0:
            signals[i] = 0.0
            continue
        
        # Long entry: bullish trend + RSI pullback
        if st_1h == 1 and st_15m == 1:
            if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                signals[i] = base_size
                in_position = True
                position_side = 1
                entry_price = price
                entry_atr = atr
                highest_since_entry = price
                tp_triggered = False
                continue
        
        # Short entry: bearish trend + RSI pullback
        elif st_1h == -1 and st_15m == -1:
            if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                signals[i] = -base_size
                in_position = True
                position_side = -1
                entry_price = price
                entry_atr = atr
                lowest_since_entry = price
                tp_triggered = False
                continue
        
        signals[i] = 0.0
    
    return signals