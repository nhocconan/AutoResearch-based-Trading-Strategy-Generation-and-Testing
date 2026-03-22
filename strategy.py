#!/usr/bin/env python3
"""
Experiment #015: 1h Volatility Compression Breakout with 4h HMA Trend Filter
Hypothesis: BTC/ETH spend most time in consolidation. Wait for BB width compression (volatility < 20th percentile),
then trade breakout in direction of 4h HMA trend. This reduces whipsaw trades during 2022 crash and 2025 range.
Key innovation: Only trade when volatility is compressed AND breaks out with RSI confirmation.
Fewer but higher-quality trades. 4h HMA prevents counter-trend entries.
Position sizing: 0.25 base, 0.35 max for strong signals, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to limit drawdown during volatile periods.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_squeeze_4h_hma_rsi_v1"
timeframe = "1h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth for squeeze detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

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

def calculate_bb_percentile(bandwidth, window=100):
    """Calculate rolling percentile of BB bandwidth for squeeze detection."""
    n = len(bandwidth)
    bb_pct = np.zeros(n)
    bb_pct[:] = np.nan
    
    for i in range(window, n):
        window_data = bandwidth[i-window:i]
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) > 0:
            bb_pct[i] = np.searchsorted(np.sort(valid_data), bandwidth[i]) / len(valid_data)
    
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, 100)
    rsi = calculate_rsi(close, 14)
    
    # Additional trend filters
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volatility squeeze detection (BB bandwidth at low percentile)
        volatility_squeeze = bb_percentile[i] < 0.25  # Bottom 25% of recent volatility
        
        # BB breakout signals (price breaks outside bands after squeeze)
        breakout_long = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        breakout_short = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # RSI momentum confirmation
        rsi_bullish = rsi[i] > 50 and rsi[i] > rsi[i-3] if i >= 3 else False
        rsi_bearish = rsi[i] < 50 and rsi[i] < rsi[i-3] if i >= 3 else False
        rsi_strong_bull = rsi[i] > 55
        rsi_strong_bear = rsi[i] < 45
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        ema_strong_bull = ema_bullish and close[i] > ema_200[i]
        ema_strong_bear = ema_bearish and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === VOLATILITY SQUEEZE BREAKOUT (primary signal) ===
        if volatility_squeeze:
            # Long: squeeze + breakout + HTF bull + RSI bullish + EMA bullish
            if breakout_long and bull_trend and rsi_bullish and ema_bullish:
                new_signal = SIZE_MAX
            # Long: squeeze + breakout + HTF bull (moderate conviction)
            elif breakout_long and bull_trend and rsi_strong_bull:
                new_signal = SIZE_BASE
            # Short: squeeze + breakout + HTF bear + RSI bearish + EMA bearish
            elif breakout_short and bear_trend and rsi_bearish and ema_bearish:
                new_signal = -SIZE_MAX
            # Short: squeeze + breakout + HTF bear (moderate conviction)
            elif breakout_short and bear_trend and rsi_strong_bear:
                new_signal = -SIZE_BASE
        
        # === TREND CONTINUATION (secondary signal - no squeeze required) ===
        else:
            # Long: HTF bull + EMA strong bull + RSI strong bull + price > BB middle
            if bull_trend and ema_strong_bull and rsi_strong_bull and close[i] > bb_sma[i]:
                new_signal = SIZE_BASE
            # Short: HTF bear + EMA strong bear + RSI strong bear + price < BB middle
            elif bear_trend and ema_strong_bear and rsi_strong_bear and close[i] < bb_sma[i]:
                new_signal = -SIZE_BASE
        
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