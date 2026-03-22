#!/usr/bin/env python3
"""
Experiment #005: 12h Multi-Timeframe Strategy with 1d HMA Trend Filter
Hypothesis: 12h timeframe captures medium-term swings while 1d HMA provides strong trend bias.
Simpler logic than failed experiments: focus on RSI pullback entries in direction of HTF trend.
Key innovation: Volume confirmation + relaxed RSI thresholds to ensure enough trades.
Position sizing: 0.25 base, 0.35 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_vol_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_sma
    ratio[vol_sma == 0] = 1.0
    return ratio

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_keltner(high, low, close, atr_period=14, mult=2.0):
    """Calculate Keltner Channel for volatility-based support/resistance."""
    atr = calculate_atr(high, low, close, atr_period)
    ema = calculate_ema(close, 20)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    ema_1d_200 = calculate_ema(df_1d['close'].values, 200)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    volume_ratio = calculate_volume_ratio(volume, 20)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    keltner_upper, keltner_lower, keltner_atr = calculate_keltner(high, low, close, 14, 2.0)
    
    # SMA for trend filter
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary filter
        bull_trend_1d = close[i] > hma_1d_21_aligned[i] and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_trend_1d = close[i] < hma_1d_21_aligned[i] and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # 12h trend confirmation
        bull_trend_12h = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        bear_trend_12h = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # Strong trend when both HTF and LTF agree
        strong_bull = bull_trend_1d and bull_trend_12h
        strong_bear = bear_trend_1d and bear_trend_12h
        
        # Volume confirmation
        vol_confirmed = volume_ratio[i] > 1.2
        
        # RSI pullback signals (relaxed for more trades)
        rsi_pullback_long = rsi_14[i] < 50 and rsi_14[i] > 35 and rsi_7[i] < rsi_7[i-1] if i >= 1 else False
        rsi_pullback_short = rsi_14[i] > 50 and rsi_14[i] < 65 and rsi_7[i] > rsi_7[i-1] if i >= 1 else False
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # MACD momentum
        macd_bull = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i >= 1 else False
        macd_bear = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i >= 1 else False
        macd_cross_up = macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1] if i >= 1 else False
        macd_cross_down = macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1] if i >= 1 else False
        
        # Donchian breakout
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Keltner touch for mean reversion
        keltner_long = close[i] < keltner_lower[i] * 1.01
        keltner_short = close[i] > keltner_upper[i] * 0.99
        
        # HMA crossover
        hma_cross_long = hma_21[i] > hma_50[i] and hma_21[i-1] <= hma_50[i-1] if i >= 1 else False
        hma_cross_short = hma_21[i] < hma_50[i] and hma_21[i-1] >= hma_50[i-1] if i >= 1 else False
        
        new_signal = 0.0
        
        # === STRONG TREND: Breakout entries with momentum ===
        if strong_bull:
            # Breakout with volume
            if breakout_long and vol_confirmed:
                new_signal = SIZE_MAX
            # RSI pullback in uptrend
            elif rsi_pullback_long and macd_bull:
                new_signal = SIZE_BASE
            # HMA crossover confirmation
            elif hma_cross_long and vol_confirmed:
                new_signal = SIZE_BASE
            # MACD cross with trend
            elif macd_cross_up and bull_trend_12h:
                new_signal = SIZE_HALF
        
        elif strong_bear:
            # Breakout with volume
            if breakout_short and vol_confirmed:
                new_signal = -SIZE_MAX
            # RSI pullback in downtrend
            elif rsi_pullback_short and macd_bear:
                new_signal = -SIZE_BASE
            # HMA crossover confirmation
            elif hma_cross_short and vol_confirmed:
                new_signal = -SIZE_BASE
            # MACD cross with trend
            elif macd_cross_down and bear_trend_12h:
                new_signal = -SIZE_HALF
        
        # === MODERATE TREND: Single timeframe confirmation ===
        elif bull_trend_1d or bull_trend_12h:
            # RSI oversold bounce
            if rsi_oversold and macd_bull:
                new_signal = SIZE_HALF
            # Keltner lower touch in uptrend
            elif keltner_long and bull_trend_1d:
                new_signal = SIZE_HALF
            # HMA cross with volume
            elif hma_cross_long and vol_confirmed:
                new_signal = SIZE_HALF
        
        elif bear_trend_1d or bear_trend_12h:
            # RSI overbought rejection
            if rsi_overbought and macd_bear:
                new_signal = -SIZE_HALF
            # Keltner upper touch in downtrend
            elif keltner_short and bear_trend_1d:
                new_signal = -SIZE_HALF
            # HMA cross with volume
            elif hma_cross_short and vol_confirmed:
                new_signal = -SIZE_HALF
        
        # === RANGE/WEAK TREND: Mean reversion only ===
        else:
            # Keltner mean reversion with RSI extreme
            if keltner_long and rsi_oversold:
                new_signal = SIZE_HALF
            elif keltner_short and rsi_overbought:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals