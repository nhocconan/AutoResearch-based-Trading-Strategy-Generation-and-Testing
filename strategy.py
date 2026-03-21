#!/usr/bin/env python3
"""
EXPERIMENT #082 - Regime Adaptive DEMA+Volume+BBW Percentile (15m+4h Proper HTF v1)
==================================================================================================
Hypothesis: Recent failures (#070-#081) show ensemble voting creates too much churn and DD.
Key insight: SIMPLER regime detection + proper HTF alignment + volume confirmation = better Sharpe.

Why this should beat current best (Sharpe=3.653):
- Use mtf_data helper for PROPER 4h alignment (critical for SOL data gaps - 46 strategies failed without)
- DEMA(8/21) faster than HMA for trend detection (less lag)
- BBW PERCENTILE (rolling 100) instead of absolute threshold (adapts to each asset's vol regime)
- Volume confirmation filter (avoid low-liquidity fakeouts)
- Discrete position sizes (0.0, ±0.25, ±0.35) to minimize fee churn
- Tighter stoploss (1.5*ATR) with faster TP (1.5R first, trail at 1R)
- Position size capped at 0.35 max (DD control)

Lessons from failures:
- #070, #071, #078: Ensemble voting → too many signal changes → fee drain + DD
- #073, #075: Proper HTF alignment improved Sharpe (use mtf_data!)
- All failures: Manual resampling breaks on SOL data gaps

Risk management:
- Max position: 0.35 (35% of capital)
- Stoploss: 1.5*ATR from entry
- Take profit: 50% at 1.5R, trail rest at 1R
- ADX filter: only trade when 4h ADX > 20 (trend confirmation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "regime_adaptive_dema_volume_bbw_pct_15m_4h_v1"
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


def calculate_dema(close, fast_period=8, slow_period=21):
    """Calculate Double Exponential Moving Average crossover signal"""
    n = len(close)
    if n < slow_period * 2:
        return np.zeros(n), np.zeros(n)
    
    # Calculate EMAs
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast_period - 1] = np.mean(close[:fast_period])
    ema_slow[slow_period - 1] = np.mean(close[:slow_period])
    
    for i in range(fast_period, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast_period + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow_period, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow_period + 1)) * (close[i] - ema_slow[i - 1])
    
    # DEMA = 2*EMA - EMA(EMA)
    dema_fast = np.zeros(n)
    dema_slow = np.zeros(n)
    
    ema_of_ema_fast = np.zeros(n)
    ema_of_ema_slow = np.zeros(n)
    
    ema_of_ema_fast[fast_period * 2 - 2] = np.mean(ema_fast[fast_period - 1:fast_period * 2 - 1])
    ema_of_ema_slow[slow_period * 2 - 2] = np.mean(ema_slow[slow_period - 1:slow_period * 2 - 1])
    
    for i in range(fast_period * 2 - 1, n):
        ema_of_ema_fast[i] = ema_of_ema_fast[i - 1] + (2.0 / (fast_period + 1)) * (ema_fast[i] - ema_of_ema_fast[i - 1])
        dema_fast[i] = 2 * ema_fast[i] - ema_of_ema_fast[i]
    
    for i in range(slow_period * 2 - 1, n):
        ema_of_ema_slow[i] = ema_of_ema_slow[i - 1] + (2.0 / (slow_period + 1)) * (ema_slow[i] - ema_of_ema_slow[i - 1])
        dema_slow[i] = 2 * ema_slow[i] - ema_of_ema_slow[i]
    
    # Crossover signal: 1 if fast > slow, -1 if fast < slow, 0 otherwise
    signal = np.zeros(n)
    for i in range(slow_period * 2 - 1, n):
        if dema_fast[i] > dema_slow[i]:
            signal[i] = 1
        elif dema_fast[i] < dema_slow[i]:
            signal[i] = -1
    
    return signal, dema_fast


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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def calculate_bbw_percentile(bbw, lookback=100):
    """Calculate BBW percentile rank (regime detection)"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i]) / lookback
        percentile[i] = rank
    
    return percentile


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    vol_ma = np.zeros(n)
    vol_ma[period - 1] = np.mean(volume[:period])
    
    for i in range(period, n):
        vol_ma[i] = vol_ma[i - 1] + (volume[i] - volume[i - period]) / period
    
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    dema_signal_15m, dema_fast_15m = calculate_dema(close, fast_period=8, slow_period=21)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=100)
    vol_ma_15m = calculate_volume_ma(volume, period=20)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # 4h indicators for trend filter
        adx_4h_raw = calculate_adx(high_4h, low_4h, close_4h, period=14)
        _, _, _, bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)
        bbw_pct_4h_raw = calculate_bbw_percentile(bbw_4h_raw, lookback=100)
        dema_signal_4h_raw, _ = calculate_dema(close_4h, fast_period=8, slow_period=21)
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        adx_4h = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
        bbw_pct_4h = align_htf_to_ltf(prices, df_4h, bbw_pct_4h_raw)
        dema_signal_4h = align_htf_to_ltf(prices, df_4h, dema_signal_4h_raw)
        
    except Exception as e:
        # Fallback if mtf_data fails (shouldn't happen in production)
        adx_4h = np.zeros(n)
        bbw_pct_4h = np.zeros(n)
        dema_signal_4h = np.zeros(n)
    
    # Generate signals with regime-adaptive logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX threshold for trend strength (4h)
    ADX_MIN = 20
    
    # BBW percentile thresholds for regime
    BBW_PCT_LOW = 0.30  # Low vol regime (trend follow)
    BBW_PCT_HIGH = 0.70  # High vol regime (mean revert)
    
    # Volume filter (avoid low liquidity)
    VOL_MIN_RATIO = 0.8
    
    # ATR stoploss multiplier (tighter than previous)
    ATR_STOP_MULT = 1.5
    ATR_TP_MULT = 1.5
    
    first_valid = max(300, 14 * 2, 20, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend filters
        adx_4h_val = adx_4h[i]
        dema_4h_signal = dema_signal_4h[i]
        bbw_pct_4h_val = bbw_pct_4h[i]
        
        # 15m entry signals
        dema_15m_signal = dema_signal_15m[i]
        rsi_val = rsi_15m[i]
        bbw_pct_15m_val = bbw_pct_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
        # Volume filter
        vol_ratio = volume[i] / vol_ma_15m[i] if vol_ma_15m[i] > 0 else 0
        
        # ADX filter (4h) - only trade when trend is strong enough
        if adx_4h_val < ADX_MIN:
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
            
            # Stoploss check (1.5*ATR)
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
                
                # Take profit check (1.5R) - reduce to half
                tp_price = prev_entry + ATR_TP_MULT * ATR_STOP_MULT * atr
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
                
                # Take profit check (1.5R) - reduce to half
                tp_price = prev_entry - ATR_TP_MULT * ATR_STOP_MULT * atr
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
        
        # Regime detection: BBW percentile determines entry type
        # Low vol regime (pct < 0.30): trend follow
        # High vol regime (pct > 0.70): mean revert
        # Mid vol: no trading
        
        # Volume confirmation
        if vol_ratio < VOL_MIN_RATIO:
            signals[i] = 0.0
            continue
        
        # 4h DEMA trend direction
        if dema_4h_signal == 1:  # Bullish 4h trend
            # Low vol regime: trend follow on 15m DEMA bullish + RSI pullback
            if bbw_pct_4h_val < BBW_PCT_LOW:
                if (dema_15m_signal == 1 and 
                    RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            # High vol regime: mean revert on 15m RSI oversold
            elif bbw_pct_4h_val > BBW_PCT_HIGH:
                if rsi_val < 35:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                    
        elif dema_4h_signal == -1:  # Bearish 4h trend
            # Low vol regime: trend follow on 15m DEMA bearish + RSI pullback
            if bbw_pct_4h_val < BBW_PCT_LOW:
                if (dema_15m_signal == -1 and 
                    RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            # High vol regime: mean revert on 15m RSI overbought
            elif bbw_pct_4h_val > BBW_PCT_HIGH:
                if rsi_val > 65:
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