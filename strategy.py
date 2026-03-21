#!/usr/bin/env python3
"""
EXPERIMENT #008 - MTF Mean Reversion + Trend Filter (1h+4h+1d v1)
==================================================================================================
Hypothesis: Mean reversion strategies have NOT been tested in recent experiments. This combines:
- 1h Bollinger Band mean reversion entries (price touching bands)
- 4h HMA trend filter (only trade with higher timeframe trend)
- 1d SMA50 regime filter (avoid counter-trend mean reversion in strong trends)
- RSI + Z-score confirmation for statistical edge
- Conservative sizing (0.25 max) with ATR stops

Why this should work:
- Mean reversion works well in ranging markets (BBW filter)
- 4h trend filter avoids catching falling knives
- Daily SMA50 prevents trading against major trend
- Lower trade frequency than 15m strategies = less fee drag
- Discrete signal levels reduce churn costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_meanrev_bb_hma_zscore_1h_4h_1d_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.ones(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    
    return upper, middle, lower, bbw


def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = atr > 0
    plus_di[mask] = 100 * pd.Series(plus_dm / atr).ewm(span=period, adjust=False).mean().values[mask]
    minus_di[mask] = 100 * pd.Series(minus_dm / atr).ewm(span=period, adjust=False).mean().values[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for mean reversion entries
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    bb_upper_1h, bb_middle_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    zscore_1h = calculate_zscore(close, period=20)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    hma_4h = calculate_hma(c_4h, period=21)
    rsi_4h = calculate_rsi(c_4h, period=14)
    
    # Align 4h indicators to 1h timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    c_1d = df_1d['close'].values
    
    sma_1d = pd.Series(c_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily SMA to 1h timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # Mean reversion thresholds
    BB_TOUCH_THRESHOLD = 0.95  # Price must be within 5% of band
    ZSCORE_ENTRY = 1.5  # Z-score threshold for entry
    ZSCORE_EXIT = 0.5  # Z-score threshold for exit
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    ADX_MIN = 15  # Minimum ADX for trend confirmation
    BBW_MIN = 0.02  # Minimum BB width to avoid extremely tight bands
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 50 * 24)  # Need daily SMA50 aligned (50 days * 24 hours)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_zscore = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = position_side[i-1]
            continue
        
        # Get aligned MTF values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        sma_1d_val = sma_1d_aligned[i] if i < len(sma_1d_aligned) else close[i]
        
        # BBW filter - avoid extremely tight bands
        if bbw_1h[i] < BBW_MIN or np.isnan(bbw_1h[i]):
            signals[i] = 0.0
            if i > 0:
                position_side[i] = 0
            continue
        
        # 4h trend filter (price vs HMA)
        trend_4h = 0
        if hma_4h_val > 0:
            if c_4h[min(i // 4, len(c_4h) - 1)] > hma_4h_val:
                trend_4h = 1
            elif c_4h[min(i // 4, len(c_4h) - 1)] < hma_4h_val:
                trend_4h = -1
        
        # Daily regime filter (price vs SMA50)
        trend_1d = 0
        if sma_1d_val > 0:
            if close[i] > sma_1d_val:
                trend_1d = 1
            elif close[i] < sma_1d_val:
                trend_1d = -1
        
        # Check existing positions first (stoploss / take profit)
        if position_side[i-1] != 0 and i > 0:
            prev_side = position_side[i-1]
            prev_entry = entry_price[i-1] if entry_price[i-1] > 0 else close[i-1]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if highest_since_entry[i-1] > 0 else prev_entry, close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if lowest_since_entry[i-1] > 0 else prev_entry, close[i])
            else:
                highest_since_entry[i] = max(highest_since_entry[i-1] if highest_since_entry[i-1] > 0 else prev_entry, close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if lowest_since_entry[i-1] > 0 else prev_entry, close[i])
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_zscore[i] = 0
                    continue
                
                # Take profit: exit when Z-score returns to mean (zscore < 0.5)
                if zscore_1h[i] > -ZSCORE_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_zscore[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_zscore[i] = 0
                    continue
                
                # Take profit: exit when Z-score returns to mean (zscore > -0.5)
                if zscore_1h[i] < ZSCORE_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_zscore[i] = 0
                    continue
            
            # Hold position
            signals[i] = signals[i-1]
            position_side[i] = position_side[i-1]
            entry_price[i] = entry_price[i-1]
            entry_zscore[i] = entry_zscore[i-1]
            continue
        
        # Entry logic: Mean reversion with trend filter
        # Only enter long when 4h trend is bullish AND price touches lower BB
        # Only enter short when 4h trend is bearish AND price touches upper BB
        
        price_vs_lower_bb = (close[i] - bb_lower_1h[i]) / (bb_upper_1h[i] - bb_lower_1h[i]) if (bb_upper_1h[i] - bb_lower_1h[i]) > 0 else 0.5
        price_vs_upper_bb = (bb_upper_1h[i] - close[i]) / (bb_upper_1h[i] - bb_lower_1h[i]) if (bb_upper_1h[i] - bb_lower_1h[i]) > 0 else 0.5
        
        # Long entry: 4h bullish, price near lower BB, RSI oversold, Z-score negative
        if (trend_4h == 1 and 
            trend_1d >= 0 and
            price_vs_lower_bb < BB_TOUCH_THRESHOLD and
            rsi_1h[i] < RSI_OVERSOLD and
            zscore_1h[i] < -ZSCORE_ENTRY and
            adx_1h[i] > ADX_MIN):
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = close[i]
            entry_zscore[i] = zscore_1h[i]
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        # Short entry: 4h bearish, price near upper BB, RSI overbought, Z-score positive
        elif (trend_4h == -1 and 
              trend_1d <= 0 and
              price_vs_upper_bb < BB_TOUCH_THRESHOLD and
              rsi_1h[i] > RSI_OVERBOUGHT and
              zscore_1h[i] > ZSCORE_ENTRY and
              adx_1h[i] > ADX_MIN):
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = close[i]
            entry_zscore[i] = zscore_1h[i]
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals