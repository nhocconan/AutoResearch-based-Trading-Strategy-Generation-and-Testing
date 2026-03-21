#!/usr/bin/env python3
"""
Hypothesis: 4h primary with KAMA adaptive trend + Supertrend direction + Bollinger regime filter.
KAMA adapts to market noise (fast in trends, slow in ranges), Supertrend gives clear directional bias.
Bollinger Band Width percentile detects regime: squeeze=range (mean reversion), expansion=trend (breakout).
1d HMA filters major trend direction. ATR(14) stoploss at 2*ATR protects capital.
SIZE=0.30 discrete levels with regime-adaptive entries should reduce whipsaw vs pure trend following.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_supertrend_regime_4h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    close_s = pd.Series(close)
    change = abs(close_s.diff(period))
    volatility = close_s.diff().abs().rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator with ATR-based bands"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction, atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth"""
    close_s = pd.Series(close)
    sma = close_s.rolling(period, min_periods=period).mean().values
    std = close_s.rolling(period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # 4h indicators - all computed before loop (Rule 8)
    # KAMA adaptive trend
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Supertrend direction
    supertrend, st_direction, atr = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Bollinger Band Width percentile (regime detector)
    bbw_percentile = pd.Series(bb_bandwidth).rolling(100, min_periods=50).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 50) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    bbw_percentile = np.nan_to_num(bbw_percentile, nan=0.5)
    
    # RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # EMA(50) for additional trend confirmation
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend: price vs 1d HMA
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime detection: Bollinger Band Width
        # Low BW = squeeze/range, High BW = expansion/trend
        regime_trend = bb_bandwidth[i] > bbw_percentile[i]  # above median = trend regime
        regime_range = bb_bandwidth[i] <= bbw_percentile[i]  # at/below median = range regime
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Local trend
        local_bullish = close[i] > ema50[i]
        local_bearish = close[i] < ema50[i]
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2 * atr[i]
            initial_stop = entry_price - 2 * atr[i]
            if close[i] < max(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2 * atr[i]
            initial_stop = entry_price + 2 * atr[i]
            if close[i] > min(trail_stop, initial_stop):
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # LONG entries
            long_score = 0
            
            # HTF alignment (mandatory)
            if htf_bullish:
                long_score += 2
            
            # Supertrend bullish
            if st_bullish:
                long_score += 1
            
            # KAMA bullish
            if kama_bullish:
                long_score += 1
            
            # Local trend
            if local_bullish:
                long_score += 1
            
            # RSI not overbought
            if rsi[i] < 70:
                long_score += 1
            
            # Trend regime preference for breakouts
            if regime_trend and long_score >= 4:
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            # Range regime - mean reversion entries (need stronger signal)
            elif regime_range and long_score >= 5 and rsi[i] < 45:
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            
            # SHORT entries
            short_score = 0
            
            # HTF alignment (mandatory)
            if htf_bearish:
                short_score += 2
            
            # Supertrend bearish
            if st_bearish:
                short_score += 1
            
            # KAMA bearish
            if kama_bearish:
                short_score += 1
            
            # Local trend
            if local_bearish:
                short_score += 1
            
            # RSI not oversold
            if rsi[i] > 30:
                short_score += 1
            
            # Trend regime preference for breakouts
            if regime_trend and short_score >= 4:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
            # Range regime - mean reversion entries (need stronger signal)
            elif regime_range and short_score >= 5 and rsi[i] > 55:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
        else:
            # Hold position
            signals[i] = signals[i-1]
    
    return signals