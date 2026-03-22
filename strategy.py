#!/usr/bin/env python3
"""
Experiment #007: 15m RSI Mean Reversion + 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 15m timeframe captures intraday mean reversion opportunities while 4h HMA 
provides trend bias to avoid counter-trend trades that fail. RSI(7) reacts faster than 
RSI(14) on 15m, catching oversold/overbought extremes within the HTF trend direction.
Volume confirmation filters false breakouts. ATR(14) stoploss at 2.5x limits drawdown.
Key innovation: Faster RSI period (7) + volume spike filter + discrete signal levels.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_4h_hma_volume_v1"
timeframe = "15m"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_std = vol_s.rolling(window=period, min_periods=period).std().values
    vol_zscore = (volume - vol_avg) / (vol_std + 1e-10)
    return vol_zscore

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_zscore = calculate_volume_spike(volume, 20)
    
    # Additional trend filters
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
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
        
        if np.isnan(rsi_7[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        
        # RSI extremes (faster period for 15m)
        rsi_oversold = rsi_7[i] < 30
        rsi_overbought = rsi_7[i] > 70
        rsi_extreme_oversold = rsi_7[i] < 20
        rsi_extreme_overbought = rsi_7[i] > 80
        
        # Bollinger Band positions
        price_below_lower = close[i] < bb_lower[i] * 1.002
        price_above_upper = close[i] > bb_upper[i] * 0.998
        price_near_sma = abs(close[i] - bb_sma[i]) < bb_sma[i] * 0.005
        
        # Volume confirmation
        volume_spike = vol_zscore[i] > 1.5  # Above average volume
        volume_normal = abs(vol_zscore[i]) < 1.0  # Normal volume for mean reversion
        
        # ATR volatility filter
        atr_ratio = atr[i] / (np.nanmean(atr[max(0, i-50):i]) + 1e-10) if i > 50 else 1.0
        vol_normal_regime = 0.5 < atr_ratio < 2.0  # Not extreme volatility
        
        new_signal = 0.0
        
        # === LONG SIGNALS (with 4h bull trend bias) ===
        if bull_trend:
            # Mean reversion long: price at lower BB + RSI oversold
            if price_below_lower and rsi_oversold and volume_normal:
                new_signal = SIZE_BASE
            # Strong mean reversion: extreme RSI + BB touch
            elif price_below_lower and rsi_extreme_oversold:
                new_signal = SIZE_MAX
            # Pullback long in uptrend: RSI dips but price > EMA21
            elif rsi_oversold and close[i] > ema_21[i] and ema_bullish:
                new_signal = SIZE_BASE
            # Volume confirmation breakout pullback
            elif rsi_oversold and volume_spike and close[i] > bb_sma[i]:
                new_signal = SIZE_BASE
        
        # === SHORT SIGNALS (with 4h bear trend bias) ===
        elif bear_trend:
            # Mean reversion short: price at upper BB + RSI overbought
            if price_above_upper and rsi_overbought and volume_normal:
                new_signal = -SIZE_BASE
            # Strong mean reversion: extreme RSI + BB touch
            elif price_above_upper and rsi_extreme_overbought:
                new_signal = -SIZE_MAX
            # Pullback short in downtrend: RSI rises but price < EMA21
            elif rsi_overbought and close[i] < ema_21[i] and ema_bearish:
                new_signal = -SIZE_BASE
            # Volume confirmation breakout pullback
            elif rsi_overbought and volume_spike and close[i] < bb_sma[i]:
                new_signal = -SIZE_BASE
        
        # === RANGE REGIME (when 4h trend is unclear) ===
        # Use pure mean reversion at BB extremes
        if abs(close[i] - hma_4h_aligned[i]) < hma_4h_aligned[i] * 0.01:
            if price_below_lower and rsi_extreme_oversold and vol_normal_regime:
                new_signal = SIZE_BASE
            elif price_above_upper and rsi_extreme_overbought and vol_normal_regime:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
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
            
            # Calculate trailing stop (2.5*ATR)
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