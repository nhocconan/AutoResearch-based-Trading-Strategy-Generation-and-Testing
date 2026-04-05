#!/usr/bin/env python3
"""
exp_7539_6d_2025_06_05
Hypothesis: 6-hour Bollinger Band squeeze breakout with 12-hour volume confirmation and 1-day trend filter.
In trending markets (ADX > 25): breakout in direction of trend.
In ranging markets (ADX <= 25): mean reversion at Bollinger Bands.
Uses Bollinger Band width to detect squeeze (low volatility) and expansion (high volatility).
Volume must confirm breakout strength. Targets 75-200 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7539_6d_2025_06_05"
timeframe = "6h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
BBW_PERIOD = 50  # for percentile
ADX_PERIOD = 14
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
        minus_dm[i] = max(high[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    dx = np.zeros_like(high)
    for i in range(period, len(high)):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = np.zeros_like(high)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Bollinger Bands and Band Width
    close_12h = df_12h['close'].values
    bb_middle = pd.Series(close_12h).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    bb_std = pd.Series(close_12h).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    bb_upper = bb_middle + (BB_STD * bb_std)
    bb_lower = bb_middle - (BB_STD * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate BBW percentile for squeeze detection
    bbw_percentile = pd.Series(bb_width).rolling(window=BBW_PERIOD, min_periods=BBW_PERIOD).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, ADX_PERIOD)
    
    # Align HTF indicators to LTF
    bb_middle_aligned = align_htf_to_ltf(prices, df_12h, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    bbw_percentile_aligned = align_htf_to_ltf(prices, df_12h, bbw_percentile)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BBW_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(bbw_percentile_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        trending = adx_1d_aligned[i] > 25  # ADX > 25 = trending
        ranging = adx_1d_aligned[i] <= 25  # ADX <= 25 = ranging
        
        # Bollinger Band conditions
        squeeze = bbw_percentile_aligned[i] < 20  # Low volatility (squeeze)
        expansion = bbw_percentile_aligned[i] > 80  # High volatility (expansion)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        long_breakout = close[i] > bb_upper_aligned[i]
        short_breakout = close[i] < bb_lower_aligned[i]
        
        # Mean reversion conditions
        long_reversion = close[i] < bb_lower_aligned[i]
        short_reversion = close[i] > bb_upper_aligned[i]
        
        # Entry conditions
        if position == 0:
            if trending and expansion and volume_confirmed:
                # Trend following: breakout in direction of price momentum
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif short_breakout:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif ranging and squeeze and volume_confirmed:
                # Mean reversion: fade extreme moves during low volatility
                if long_reversion:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                elif short_reversion:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals