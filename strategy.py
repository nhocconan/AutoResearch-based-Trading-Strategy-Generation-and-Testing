#!/usr/bin/env python3
"""
Experiment #019: 15m MACD Volume Breakout + 1h HMA Trend Filter
Hypothesis: MACD momentum with volume confirmation generates more reliable signals than RSI mean-reversion on 15m.
Combined with 1h HMA trend filter to avoid counter-trend trades. Volume spike (1.5x avg) confirms breakout validity.
This should generate MORE trades than CRSI strategies (which had 0 trades in exp#014) while maintaining quality.
Timeframe: 15m (REQUIRED), HTF: 1h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Volume confirmation reduces false breakouts, MACD histogram crossover provides timely entries.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_macd_vol_1h_hma_trend_v1"
timeframe = "15m"
leverage = 1.0

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes relative to rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[np.isnan(vol_ratio)] = 1.0
    vol_ratio[np.isinf(vol_ratio)] = 1.0
    return vol_ratio

def calculate_ema(close, period):
    """Calculate exponential moving average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    atr = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_spike(volume, period=20, threshold=1.5)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or np.isnan(histogram[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]):
            signals[i] = 0.0
            continue
        
        # 1h trend bias (HTF)
        bull_trend = close[i] > hma_1h_aligned[i]
        bear_trend = close[i] < hma_1h_aligned[i]
        
        # MACD momentum signals
        macd_bull_cross = histogram[i] > 0 and histogram[i-1] <= 0  # Histogram crosses above zero
        macd_bear_cross = histogram[i] < 0 and histogram[i-1] >= 0  # Histogram crosses below zero
        macd_positive = histogram[i] > 0
        macd_negative = histogram[i] < 0
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.3  # Volume 30% above average
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        # Donchian breakout
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: MACD bull cross + volume confirmed + 1h bull trend
        if macd_bull_cross and volume_confirmed and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: MACD positive + EMA bullish + 1h bull trend
        elif macd_positive and ema_bullish and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: Donchian breakout high + volume + 1h bull trend
        elif breakout_high and volume_confirmed and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: MACD bear cross + volume confirmed + 1h bear trend
        if macd_bear_cross and volume_confirmed and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: MACD negative + EMA bearish + 1h bear trend
        elif macd_negative and ema_bearish and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: Donchian breakout low + volume + 1h bear trend
        elif breakout_low and volume_confirmed and bear_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals