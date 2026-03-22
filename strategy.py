#!/usr/bin/env python3
"""
Experiment #004: 4h Mean Reversion + 1d Trend Filter + RSI/Bollinger + ATR Stop
Hypothesis: 4h timeframe captures multi-day swings while 1d HMA provides regime bias.
Mean reversion works better than pure trend in bear/range markets (2022, 2025).
Long: 1d bullish + 4h RSI<30 + price<BB_lower + Supertrend flip.
Short: 1d bearish + 4h RSI>70 + price>BB_upper + Supertrend flip.
ADX>20 filter ensures we trade when volatility supports moves. 2*ATR stoploss.
Conservative sizing (0.25) controls DD during 2022 crash (-77%).
Multiple entry paths ensure >=10 trades per symbol. Timeframe: 4h (REQUIRED), HTF: 1d.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_meanrev_1d_hma_rsi_bb_supertrend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - trend following with ATR bands."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    percent_b = (close - lower) / (upper - lower)
    percent_b = np.where(np.isnan(percent_b), 0.5, percent_b)
    return upper, lower, sma, bandwidth, percent_b

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / std
    zscore = np.where(np.isnan(zscore), 0.0, zscore)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_fast = calculate_hma(df_1d['close'].values, 10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1d_fast_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_fast)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid, bb_bw, bb_pct = calculate_bollinger(close, 20, 2.0)
    zscore = calculate_zscore(close, 20)
    
    # 4h HMA for additional confirmation
    hma_4h = calculate_hma(close, 21)
    hma_4h_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary regime filter
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        htf_hma_golden = hma_1d_fast_aligned[i] > hma_1d_aligned[i]
        htf_hma_death = hma_1d_fast_aligned[i] < hma_1d_aligned[i]
        
        # 4h Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip signals (strong reversal signal)
        st_flip_long = False
        st_flip_short = False
        if i > 0 and not np.isnan(st_direction[i]) and not np.isnan(st_direction[i-1]):
            st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
            st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # 4h HMA trend
        hma_4h_bullish = close[i] > hma_4h[i]
        hma_4h_bearish = close[i] < hma_4h[i]
        hma_4h_rising = hma_4h[i] > hma_4h[i-1] if i > 0 else False
        hma_4h_falling = hma_4h[i] < hma_4h[i-1] if i > 0 else False
        
        # Fast HMA crossover on 4h
        fast_above_slow = hma_4h_fast[i] > hma_4h[i]
        fast_below_slow = hma_4h_fast[i] < hma_4h[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 20
        
        # RSI zones - mean reversion signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # Fast RSI confirmation
        rsi_fast_oversold = rsi_fast[i] < 30
        rsi_fast_overbought = rsi_fast[i] > 70
        
        # Bollinger position
        bb_low = close[i] <= bb_lower[i]
        bb_high = close[i] >= bb_upper[i]
        bb_pct_low = bb_pct[i] < 0.1
        bb_pct_high = bb_pct[i] > 0.9
        
        # Z-score extremes
        zscore_low = zscore[i] < -1.5
        zscore_high = zscore[i] > 1.5
        
        # BB squeeze (low volatility before breakout)
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i], 20) if i > 100 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (mean reversion in bullish regime) ===
        
        # Path 1: 1d bullish + 4h RSI oversold + price at BB lower + ST turning bullish
        if htf_bullish and rsi_oversold and bb_low and st_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1d bullish + 4h RSI extreme + Z-score low + ST flip long
        elif htf_bullish and rsi_extreme_oversold and zscore_low and st_flip_long:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d HMA golden + 4h HMA rising + RSI rising from oversold
        elif htf_hma_golden and hma_4h_rising and rsi_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1d bullish + 4h BB pct low + ADX weak (range) + ST bullish
        elif htf_bullish and bb_pct_low and trend_weak and st_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 5: 1d bullish + 4h HMA golden + fast HMA crossover up + RSI > 40
        elif htf_bullish and htf_hma_golden and fast_above_slow and rsi[i] > 40 and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 6: 1d bullish + 4h RSI fast oversold + price < 4h HMA (pullback)
        elif htf_bullish and rsi_fast_oversold and close[i] < hma_4h[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (mean reversion in bearish regime) ===
        
        # Path 1: 1d bearish + 4h RSI overbought + price at BB upper + ST turning bearish
        if htf_bearish and rsi_overbought and bb_high and st_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1d bearish + 4h RSI extreme + Z-score high + ST flip short
        elif htf_bearish and rsi_extreme_overbought and zscore_high and st_flip_short:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d HMA death + 4h HMA falling + RSI falling from overbought
        elif htf_hma_death and hma_4h_falling and rsi_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1d bearish + 4h BB pct high + ADX weak (range) + ST bearish
        elif htf_bearish and bb_pct_high and trend_weak and st_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 1d bearish + 4h HMA death + fast HMA crossover down + RSI < 60
        elif htf_bearish and htf_hma_death and fast_below_slow and rsi[i] < 60 and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 6: 1d bearish + 4h RSI fast overbought + price > 4h HMA (rally)
        elif htf_bearish and rsi_fast_overbought and close[i] > hma_4h[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 4h timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 4h timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals