#!/usr/bin/env python3
"""
Experiment #160: 4h Keltner Channel Breakout with Volume + Daily HMA Trend Filter
Hypothesis: Keltner Channels (EMA + ATR bands) provide cleaner volatility breakouts than
Bollinger Bands in crypto markets. Volume confirmation (1.5x avg) filters false breakouts.
Daily HMA(21) provides major trend bias. ADX(14) > 15 confirms trend strength without
being too restrictive (ADX>25 rarely triggers). Entry thresholds kept loose to ensure
sufficient trades - this was the #1 failure mode in experiments #148, #152, #156.
Position sizing: 0.25 entry, 0.125 at 2R profit. ATR stoploss at 2.5*ATR. This targets
volatility expansion breakouts which work in both trending and range markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_volume_daily_hma_adx_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate DM and TR
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_keltner_channels(high, low, close, ema_period=20, atr_period=14, multiplier=2.0):
    """Calculate Keltner Channels (EMA +/- ATR*multiplier)."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, lower, ema

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.maximum(vol_avg, 1e-10)
    return vol_ratio > threshold

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    kc_upper, kc_lower, kc_ema = calculate_keltner_channels(high, low, close, 20, 14, 2.0)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Additional trend filters
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Daily trend filter (major trend direction)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h trend filter
        trend_bullish = ema_20[i] > ema_50[i]
        trend_bearish = ema_20[i] < ema_50[i]
        
        # ADX trend strength (loose threshold for more trades)
        trend_strong = adx[i] > 15
        trend_weak = adx[i] <= 15
        
        # Keltner Channel breakout signals
        kc_breakout_up = close[i] > kc_upper[i]
        kc_breakout_down = close[i] < kc_lower[i]
        kc_revert_to_ema = kc_lower[i] < close[i] < kc_upper[i]
        
        # Volume confirmation
        volume_confirmed = vol_spike[i]
        
        # DI crossover for momentum
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # LONG ENTRY: Keltner upper breakout + volume + trend confirmation
        if kc_breakout_up:
            if daily_bullish and volume_confirmed:
                # Strong long: daily bullish + volume spike
                new_signal = SIZE_ENTRY
            elif trend_bullish and di_bullish:
                # Moderate long: 4h trend + DI bullish
                new_signal = SIZE_ENTRY
            elif trend_weak and kc_revert_to_ema and close[i] > kc_ema[i]:
                # Range market: price above Keltner EMA
                new_signal = SIZE_ENTRY * 0.8
        
        # SHORT ENTRY: Keltner lower breakout + volume + trend confirmation
        elif kc_breakout_down:
            if daily_bearish and volume_confirmed:
                # Strong short: daily bearish + volume spike
                new_signal = -SIZE_ENTRY
            elif trend_bearish and di_bearish:
                # Moderate short: 4h trend + DI bearish
                new_signal = -SIZE_ENTRY
            elif trend_weak and kc_revert_to_ema and close[i] < kc_ema[i]:
                # Range market: price below Keltner EMA
                new_signal = -SIZE_ENTRY * 0.8
        
        # TREND FOLLOWING: EMA crossover with ADX confirmation
        if new_signal == 0.0:
            if trend_bullish and ema_20[i-1] <= ema_50[i-1] and adx[i] > 15:
                if daily_bullish or di_bullish:
                    new_signal = SIZE_ENTRY
            
            elif trend_bearish and ema_20[i-1] >= ema_50[i-1] and adx[i] > 15:
                if daily_bearish or di_bearish:
                    new_signal = -SIZE_ENTRY
        
        # MEAN REVERSION: Price at Keltner extremes in weak trend
        if new_signal == 0.0 and trend_weak:
            # Long when price touches lower band in weak trend
            if close[i] < kc_lower[i] * 1.005 and close[i-1] >= kc_lower[i-1]:
                new_signal = SIZE_ENTRY * 0.6
            
            # Short when price touches upper band in weak trend
            elif close[i] > kc_upper[i] * 0.995 and close[i-1] <= kc_upper[i-1]:
                new_signal = -SIZE_ENTRY * 0.6
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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