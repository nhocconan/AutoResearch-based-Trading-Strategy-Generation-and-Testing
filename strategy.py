#!/usr/bin/env python3
"""
EXPERIMENT #001 - Supertrend + RSI Pullback with 4h Trend Filter (15m)
=======================================================================
Hypothesis: 15m Supertrend captures short-term momentum while 4h HMA(21)
ensures we trade in direction of higher timeframe trend. RSI pullback
entries (RSI < 45 for longs, > 55 for shorts) avoid chasing breakouts.
ADX(14) > 25 filters for strong trending conditions. ATR trailing stop
at 2.5*ATR protects capital during reversals.

Key features:
- Primary TF: 15m (faster entries than 1h/4h strategies)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: Supertrend flip + RSI pullback in trend direction
- Filter: ADX(14) > 25 for trend strength
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels (max 0.35)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4h_filter_15m_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    # Wilder's smoothing (RMA)
    atr = np.zeros(n)
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend_dir = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            supertrend_dir[i] = 0
            continue
        
        mid = (high[i] + low[i]) / 2
        upper_band = mid + multiplier * atr[i]
        lower_band = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            supertrend_dir[i] = -1
        else:
            # Determine direction
            if close[i] > supertrend[i-1]:
                supertrend_dir[i] = 1
                supertrend[i] = lower_band
            elif close[i] < supertrend[i-1]:
                supertrend_dir[i] = -1
                supertrend[i] = upper_band
            else:
                supertrend_dir[i] = supertrend_dir[i-1]
                if supertrend_dir[i] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
    
    return supertrend, supertrend_dir


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Wilder's smoothing (RMA)
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    avg_gain[period] = gain.iloc[1:period+1].mean()
    avg_loss[period] = loss.iloc[1:period+1].mean()
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
    
    rs = np.zeros(len(close))
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(len(close))
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[:period+1] = np.nan
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    # Calculate ATR for TR
    atr = calculate_atr(high, low, close, period)
    
    # Smooth +DM and -DM using Wilder's RMA
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initialize at period
    plus_di[period] = (plus_dm[:period+1].sum() / atr[period]) * 100 if atr[period] > 0 else 0
    minus_di[period] = (minus_dm[:period+1].sum() / atr[period]) * 100 if atr[period] > 0 else 0
    
    for i in range(period + 1, n):
        if atr[i] > 0:
            plus_di[i] = (plus_di[i-1] * (period - 1) + (plus_dm[i] / atr[i]) * 100) / period
            minus_di[i] = (minus_di[i-1] * (period - 1) + (minus_dm[i] / atr[i]) * 100) / period
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = dx[period:period*2].mean()
    for i in range(period * 2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    adx[:period*2] = np.nan
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 15m indicators
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Maximum position size
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend_dir[i]) or
            np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Supertrend direction
        st_direction = supertrend_dir[i]
        st_prev = supertrend_dir[i-1] if i > 0 else 0
        
        # ADX trend strength filter (must be > 25 for trending market)
        adx_strong = adx[i] > 25
        
        # RSI pullback filter
        # For longs: RSI < 50 (pullback in uptrend)
        # For shorts: RSI > 50 (pullback in downtrend)
        rsi_long_valid = rsi[i] < 50 and rsi[i] > 30  # Not oversold, but pulled back
        rsi_short_valid = rsi[i] > 50 and rsi[i] < 70  # Not overbought, but pulled back
        
        # Supertrend flip detection (entry signal)
        st_flip_long = (st_direction == 1 and st_prev == -1)  # Bullish flip
        st_flip_short = (st_direction == -1 and st_prev == 1)  # Bearish flip
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: 4h uptrend + Supertrend flip long + ADX strong + RSI pullback
        if hma_trend == 1 and st_flip_long and adx_strong and rsi_long_valid:
            target_signal = BASE_SIZE
        
        # Short entry: 4h downtrend + Supertrend flip short + ADX strong + RSI pullback
        elif hma_trend == -1 and st_flip_short and adx_strong and rsi_short_valid:
            target_signal = -BASE_SIZE
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Check if we're flipping direction (close existing first)
                if position_side != 0 and np.sign(target_signal) != position_side:
                    # Close existing position first (signal goes to 0)
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                else:
                    # New entry or add to position
                    signals[i] = target_signal
                    if position_side == 0:
                        position_side = 1 if target_signal > 0 else -1
                        highest_since_entry = close[i]
                        lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = BASE_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals