#!/usr/bin/env python3
"""
Experiment #254: 30m KAMA Adaptive Trend + 4h/1d HMA Filter + Donchian Breakout
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better 
than EMA/HMA in choppy 30m markets. Combined with 4h HMA trend bias and 1d HMA macro 
filter, this should reduce whipsaws. Donchian breakout provides clear entry triggers.
MACD histogram confirms momentum. Volume ratio adds conviction. This differs from 
failed RSI/Fisher strategies by using adaptive trend following instead of mean reversion.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 with fewer whipsaws in 2022 crash and 2025 bear market.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_donchian_4h_1d_hma_macd_volume_atr_v1"
timeframe = "30m"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    Efficiency Ratio (ER) measures trendiness vs noise.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate price change and noise
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.abs(close[:er_period] - close[0])
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
    noise[er_period:] = np.abs(np.diff(close, prepend=close[0]))
    noise_s = pd.Series(noise).rolling(window=er_period, min_periods=er_period).sum().values
    
    # Efficiency Ratio (0 = noise, 1 = pure trend)
    er = np.where(noise_s > 0, price_change / noise_s, 0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for breakout/momentum detection
    prev_macd_hist = np.roll(macd_hist, 1)
    prev_macd_hist[0] = macd_hist[0]
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    prev_donchian_upper[0] = donchian_upper[0]
    prev_donchian_lower[0] = donchian_lower[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    
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
        # HTF trend filters (looser to ensure trades)
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA adaptive trend
        kama_bullish = close[i] > kama[i] and kama[i] > prev_kama[i]
        kama_bearish = close[i] < kama[i] and kama[i] < prev_kama[i]
        
        # MACD momentum signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > prev_macd_hist[i]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < prev_macd_hist[i]
        macd_cross_up = prev_macd_hist[i] <= 0 and macd_hist[i] > 0
        macd_cross_down = prev_macd_hist[i] >= 0 and macd_hist[i] < 0
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # Donchian breakout detection
        breakout_long = close[i] > prev_donchian_upper[i]
        breakout_short = close[i] < prev_donchian_lower[i]
        
        # Price position in channel
        above_mid = close[i] > donchian_mid[i]
        below_mid = close[i] < donchian_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Breakout long with 4h trend and momentum (primary entry)
        if breakout_long:
            if htf_4h_bullish and (macd_bullish or vol_bullish):
                new_signal = SIZE_ENTRY
            elif htf_1d_bullish and kama_bullish:
                new_signal = SIZE_ENTRY
        
        # MACD cross up with trend confirmation
        elif macd_cross_up:
            if htf_4h_bullish and above_mid:
                new_signal = SIZE_ENTRY
            elif htf_1d_bullish and kama_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # KAMA trend follow with momentum
        elif kama_bullish and above_mid:
            if htf_4h_bullish and macd_bullish:
                new_signal = SIZE_ENTRY
            elif htf_1d_bullish and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Breakout short with 4h trend and momentum (primary entry)
        if breakout_short:
            if htf_4h_bearish and (macd_bearish or vol_bearish):
                new_signal = -SIZE_ENTRY
            elif htf_1d_bearish and kama_bearish:
                new_signal = -SIZE_ENTRY
        
        # MACD cross down with trend confirmation
        elif macd_cross_down:
            if htf_4h_bearish and below_mid:
                new_signal = -SIZE_ENTRY
            elif htf_1d_bearish and kama_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # KAMA trend follow with momentum
        elif kama_bearish and below_mid:
            if htf_4h_bearish and macd_bearish:
                new_signal = -SIZE_ENTRY
            elif htf_1d_bearish and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
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