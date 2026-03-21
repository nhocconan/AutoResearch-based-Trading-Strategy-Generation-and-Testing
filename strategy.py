#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF KAMA+Stoch+ADX+Volume (30m+4h+1d v1)
==================================================================================================
Hypothesis: 30m primary with 4h KAMA trend + 1d ADX regime + Stochastic pullback entries + Volume.
This differs from current best (mtf_hma_rsi_zscore_v1) by:
- KAMA instead of HMA (adapts to volatility better, less whipsaw in chop)
- Stochastic instead of RSI (faster momentum, different entry timing)
- ADX filter on daily (avoid weak trends entirely)
- Volume confirmation (entries need conviction)
- 30m timeframe (between 15m noise and 1h slowness)

Why this should work:
- KAMA's Efficiency Ratio adapts smoothing based on market noise (Kaufman 1998)
- Stochastic %K/%D cross gives cleaner entry signals than RSI levels
- Daily ADX > 25 filters out 60%+ of choppy periods
- Volume spike confirms institutional participation
- 30m captures intraday moves that 1h misses, cleaner than 15m
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_stoch_adx_volume_30m_4h_1d_v1"
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


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (noise vs trend)
    """
    n = len(close)
    if n < er_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period - 1] = close[er_period - 1]
    
    for i in range(er_period, n):
        # Efficiency Ratio: net change / sum of absolute changes
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if sum_changes > 0:
            er = net_change / sum_changes
        else:
            er = 0
        
        # Smoothing Constant
        sc = (er * (2 / (fast_sc + 1) - 2 / (slow_sc + 1)) + 2 / (slow_sc + 1)) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator %K and %D"""
    n = len(close)
    if n < k_period:
        return np.zeros(n), np.zeros(n)
    
    lowest_low = pd.Series(low).rolling(window=k_period, min_periods=k_period).min().values
    highest_high = pd.Series(high).rolling(window=k_period, min_periods=k_period).max().values
    
    stoch_k = np.zeros(n)
    for i in range(n):
        if highest_high[i] > lowest_low[i]:
            stoch_k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
        else:
            stoch_k[i] = 50
    
    stoch_d = pd.Series(stoch_k).rolling(window=d_period, min_periods=d_period).mean().values
    
    return stoch_k, stoch_d


def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    # First smooth
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100 * np.sum(plus_dm[1:i+1]) / np.sum(tr[1:i+1]) if np.sum(tr[1:i+1]) > 0 else 0
            minus_di[i] = 100 * np.sum(minus_dm[1:i+1]) / np.sum(tr[1:i+1]) if np.sum(tr[1:i+1]) > 0 else 0
        else:
            plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / np.sum(tr[max(1, i-period+1):i+1]) if np.sum(tr[max(1, i-period+1):i+1]) > 0 else 0
            minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / np.sum(tr[max(1, i-period+1):i+1]) if np.sum(tr[max(1, i-period+1):i+1]) > 0 else 0
        
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # ADX is smoothed DX
    adx[period * 2 - 1] = np.mean(dx[period:period*2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    stoch_k_30m, stoch_d_30m = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    kama_30m = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    volume_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h KAMA for adaptive trend
        kama_4h = calculate_kama(c_4h, er_period=10, fast_sc=2, slow_sc=30)
        
        # Align 4h KAMA to 30m timeframe
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        
    except Exception:
        kama_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for regime filter
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        
        # Daily ADX for trend strength filter
        adx_1d = calculate_adx(h_1d, l_1d, c_1d, period=14)
        
        # Align daily ADX to 30m timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
        
    except Exception:
        adx_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # Stochastic thresholds for pullback entries
    STOC_LONG_LOW = 20
    STOC_LONG_HIGH = 50
    STOC_SHORT_LOW = 50
    STOC_SHORT_HIGH = 80
    
    # ADX minimum for trend strength (avoid choppy markets)
    ADX_MIN = 25
    
    # Volume multiplier for confirmation
    VOLUME_MULT = 1.5
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 3, 30, 10 * 3)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(stoch_k_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        adx_1d_val = adx_1d_aligned[i] if i < len(adx_1d_aligned) else 0
        
        # Volume filter (avoid low volume periods)
        volume_ratio = volume[i] / volume_sma_30m[i] if volume_sma_30m[i] > 0 else 0
        
        # ADX filter - only trade when daily trend is strong
        if adx_1d_val < ADX_MIN:
            signals[i] = 0.0
            if i > 0 and position_side[i - 1] != 0:
                position_side[i] = 0
                signals[i] = 0.0
            else:
                position_side[i] = 0
            continue
        
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
        
        # Entry logic: 4h KAMA trend + 1d ADX regime + 30m Stochastic pullback + Volume
        price = close[i]
        
        # Determine 4h trend direction (price vs KAMA)
        trend_4h = 0
        if kama_4h_val > 0:
            if price > kama_4h_val:
                trend_4h = 1
            elif price < kama_4h_val:
                trend_4h = -1
        
        # Volume confirmation
        volume_confirmed = volume_ratio >= VOLUME_MULT
        
        if trend_4h == 1:  # Bullish trend on 4h
            # Stochastic pullback (oversold but crossing up)
            # Volume confirmation
            if (STOC_LONG_LOW <= stoch_k_30m[i] <= STOC_LONG_HIGH and
                stoch_k_30m[i] > stoch_d_30m[i] and
                volume_confirmed):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # Stochastic pullback (overbought but crossing down)
            # Volume confirmation
            if (STOC_SHORT_LOW <= stoch_k_30m[i] <= STOC_SHORT_HIGH and
                stoch_k_30m[i] < stoch_d_30m[i] and
                volume_confirmed):
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