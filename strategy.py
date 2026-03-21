#!/usr/bin/env python3
"""
EXPERIMENT #016 - MTF DEMA+Stochastic+Volume+ATR Dynamic Sizing (1h+4h v1)
==================================================================================================
Hypothesis: Current best #004 uses 15m/1h/4h with Supertrend+MACD+RSI. This experiment tries:
- 4h DEMA trend (faster than HMA, more responsive than SMA)
- 1h Stochastic entry (different momentum measure than RSI)
- Volume confirmation (proven in #009, #012 kept strategies)
- ATR-based dynamic position sizing (reduce size in high volatility)
- ADX filter for trend strength validation

Why this should beat #004 (Sharpe=3.653):
- 4h trend is more stable than 1h trend (less whipsaw)
- Stochastic has better overbought/oversold detection than RSI
- Volume confirmation filters false breakouts
- Dynamic sizing reduces risk during high volatility periods
- Based on lessons from #009 (volume worked) and #012 (kept strategy)

Key differences from #040:
- Uses 4h trend instead of 1h trend (slower, more stable)
- Uses Stochastic instead of RSI (different momentum)
- Uses DEMA instead of HMA (faster response)
- Dynamic position sizing based on ATR volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_dema_stochastic_volume_atr_1h_4h_v1"
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


def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False).mean().values
    
    dema = 2 * ema1 - ema2
    dema[:period] = 0
    
    return dema


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    n = len(close)
    if n < k_period + d_period:
        return np.zeros(n), np.zeros(n)
    
    k_percent = np.zeros(n)
    d_percent = np.zeros(n)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        
        if highest_high > lowest_low:
            k_percent[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            k_percent[i] = 50
    
    d_percent = pd.Series(k_percent).rolling(window=d_period, min_periods=d_period).mean().values
    d_percent[:d_period + k_period - 2] = 0
    
    return k_percent, d_percent


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
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
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
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


def calculate_volume_sma(volume, period=20):
    """Calculate Volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    stoch_k_1h, stoch_d_1h = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    adx_1h = calculate_adx(high, low, close, period=14)
    volume_sma_1h = calculate_volume_sma(volume, period=20)
    dema_1h = calculate_dema(close, period=21)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        volume_4h = df_4h['volume'].values
        
        # 4h indicators for trend
        dema_4h_raw = calculate_dema(close_4h, period=21)
        adx_4h_raw = calculate_adx(high_4h, low_4h, close_4h, period=14)
        atr_4h_raw = calculate_atr(high_4h, low_4h, close_4h, period=14)
        volume_sma_4h_raw = calculate_volume_sma(volume_4h, period=20)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        dema_4h = align_htf_to_ltf(prices, df_4h, dema_4h_raw)
        adx_4h = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
        atr_4h = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
        volume_sma_4h = align_htf_to_ltf(prices, df_4h, volume_sma_4h_raw)
    except Exception:
        # Fallback if mtf_data fails
        dema_4h = np.zeros(n)
        adx_4h = np.zeros(n)
        atr_4h = np.zeros(n)
        volume_sma_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_DYNAMIC_MIN = 0.20
    
    # Stochastic thresholds for entry
    STOCH_LONG_MIN = 20
    STOCH_LONG_MAX = 50
    STOCH_SHORT_MIN = 50
    STOCH_SHORT_MAX = 80
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.2
    
    # ATR-based dynamic sizing
    ATR_BASELINE = np.percentile(atr_1h[100:], 50) if len(atr_1h) > 100 else np.mean(atr_1h[100:])
    
    first_valid = max(200, 14 * 2, 20, 28)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(stoch_k_1h[i]) or np.isnan(adx_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 0
        if dema_4h[i] > 0 and close[i] > dema_4h[i]:
            trend_4h = 1
        elif dema_4h[i] > 0 and close[i] < dema_4h[i]:
            trend_4h = -1
        
        adx_4h_val = adx_4h[i]
        atr_4h_val = atr_4h[i]
        
        # 1h indicators
        stoch_k = stoch_k_1h[i]
        stoch_d = stoch_d_1h[i]
        adx_1h_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol = volume[i]
        vol_sma = volume_sma_1h[i]
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Volume confirmation - avoid low volume periods
        if vol_sma > 0 and vol < vol_sma * 0.8:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Trend filter - 4h DEMA must agree with price direction
        if trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
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
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
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
        
        # Dynamic position sizing based on ATR volatility
        if ATR_BASELINE > 0:
            atr_ratio = ATR_BASELINE / atr
            position_size = SIZE_FULL * min(max(atr_ratio, 0.5), 1.5)
            position_size = max(SIZE_DYNAMIC_MIN, min(0.40, position_size))
        else:
            position_size = SIZE_FULL
        
        # Entry logic: 4h DEMA trend + 1h Stochastic entry + Volume confirmation
        if trend_4h == 1:  # Bullish trend on 4h
            if (STOCH_LONG_MIN <= stoch_k <= STOCH_LONG_MAX and 
                stoch_k > stoch_d):  # Stochastic bullish cross in oversold zone
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            if (STOCH_SHORT_MIN <= stoch_k <= STOCH_SHORT_MAX and 
                stoch_k < stoch_d):  # Stochastic bearish cross in overbought zone
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals