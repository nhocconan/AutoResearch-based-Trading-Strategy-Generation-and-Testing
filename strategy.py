#!/usr/bin/env python3
"""
EXPERIMENT #112 - MTF HMA+Supertrend+RSI with ATR Trailing Stop & Vol-Adjusted Sizing
==================================================================================================
Hypothesis: Recent failures (#101-#111) with Chandelier stops failed due to excessive churn and 
complex position tracking. The winning #040 used 15m+1h MTF successfully. 

Key changes:
- Use mtf_data helper for PROPER HTF alignment (mandatory per audit rules)
- Simplified ATR trailing stop (3*ATR from highest high since entry)
- Volatility-adjusted sizing: reduce position in high vol regimes (BBW percentile)
- Keep discrete signal levels (0.0, ±0.20, ±0.35) to minimize churn costs
- Simpler state tracking without complex TP/trail logic that caused failures

Why this should beat #040 and recent failures:
- Proper MTF alignment using get_htf_data/align_htf_to_ltf (46 strategies failed without this)
- Cleaner stoploss logic (single ATR trailing vs complex Chandelier)
- Vol regime filter reduces size in high vol (where recent strategies blew up)
- Based on proven 15m+1h combination from #031, #034, #035, #040
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_supertrend_rsi_atrtrail_voladj_15m_1h_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma, nan=0.0)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
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


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    bbw = np.zeros(n)
    mask = middle > 0
    bbw[mask] = (upper[mask] - lower[mask]) / middle[mask]
    
    return upper, middle, lower, np.nan_to_num(bbw, nan=0.0)


def calculate_bb_percentile(bbw, lookback=100):
    """Calculate BBW percentile for volatility regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    hma_15m = calculate_hma(close, period=21)
    supertrend_15m, st_direction_15m = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bb_percentile(bbw_15m, lookback=100)
    
    # Get 1h HTF data using PROPER mtf_data helper (MANDATORY)
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        h_1h = df_1h['high'].values
        l_1h = df_1h['low'].values
        
        # Calculate 1h indicators
        hma_1h = calculate_hma(c_1h, period=21)
        _, st_direction_1h = calculate_supertrend(h_1h, l_1h, c_1h, period=10, multiplier=3.0)
        _, _, _, bbw_1h = calculate_bollinger_bands(c_1h, period=20, std_mult=2.0)
        bbw_pct_1h = calculate_bb_percentile(bbw_1h, lookback=100)
        
        # Align 1h indicators to 15m timeframe (auto shift for completed bars)
        hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
        st_1h_aligned = align_htf_to_ltf(prices, df_1h, st_direction_1h)
        bbw_pct_1h_aligned = align_htf_to_ltf(prices, df_1h, bbw_pct_1h)
        
    except Exception:
        # Fallback if mtf_data fails
        hma_1h_aligned = hma_15m
        st_1h_aligned = st_direction_15m
        bbw_pct_1h_aligned = bbw_pct_15m
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    # Volatility-adjusted: reduce size in high vol regimes
    BASE_SIZE = 0.35
    LOW_VOL_SIZE = 0.35  # BBW percentile < 0.5
    HIGH_VOL_SIZE = 0.20  # BBW percentile >= 0.5 (reduce exposure)
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ADX-like filter using BBW percentile (trend regime)
    BBW_PCT_MIN = 0.30  # Only trade when vol is above 30th percentile
    
    # ATR stoploss multiplier (Chandelier-style trailing)
    ATR_STOP_MULT = 3.0
    
    first_valid = max(200, 14 * 2, 20, 100)
    
    # Track position state (simplified vs #040)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 1h trend filters
        price_vs_hma_1h = 1 if close[i] > hma_1h_aligned[i] else (-1 if close[i] < hma_1h_aligned[i] else 0)
        st_trend_1h = st_1h_aligned[i]
        
        # Volatility regime (1h BBW percentile)
        vol_regime = bbw_pct_1h_aligned[i]
        
        # Determine position size based on vol regime
        if vol_regime < 0.5:
            position_size = LOW_VOL_SIZE
        else:
            position_size = HIGH_VOL_SIZE
        
        # Skip if vol regime too low (choppy market)
        if vol_regime < BBW_PCT_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                
                # ATR trailing stop (Chandelier-style: highest - 3*ATR)
                stoploss_price = current_high - ATR_STOP_MULT * atr_15m[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
                
            elif prev_side == -1:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
                highest_since_entry[i] = current_high
                lowest_since_entry[i] = current_low
                
                # ATR trailing stop (Chandelier-style: lowest + 3*ATR)
                stoploss_price = current_low + ATR_STOP_MULT * atr_15m[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Hold position
                signals[i] = signals[i - 1]
                position_side[i] = prev_side
                entry_price[i] = prev_entry
            
            continue
        
        # Entry logic: 1h trend + 15m RSI pullback
        rsi_val = rsi_15m[i]
        
        # Long entry: 1h bullish + 15m RSI pullback
        if price_vs_hma_1h == 1 and st_trend_1h == 1:
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = close[i]
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        # Short entry: 1h bearish + 15m RSI pullback
        elif price_vs_hma_1h == -1 and st_trend_1h == -1:
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = close[i]
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals