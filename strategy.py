#!/usr/bin/env python3
"""
Experiment #032: 30m RSI Mean Reversion + 4h HMA Trend + Vol Regime
Hypothesis: 30m timeframe captures intraday mean reversion while 4h HMA filters counter-trend trades.
Volatility regime (ATR ratio) identifies expansion phases where mean reversion works best.
Relaxed RSI thresholds (35/65) ensure sufficient trade frequency (>10 trades/symbol).
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop to limit drawdown in 2022-style crashes.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Key innovation: ATR ratio > 1.3 indicates vol expansion = better mean reversion opportunities.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_4h_hma_vol_regime_v2"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's smoothing."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    
    # Separate gains and losses
    gains = np.zeros(n - 1)
    losses = np.zeros(n - 1)
    gains[delta > 0] = delta[delta > 0]
    losses[delta < 0] = -delta[delta < 0]
    
    # Initial average using simple mean
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Set first valid RSI
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # Wilder's smoothing for remaining values
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[:] = np.nan
    
    if n < period:
        return atr
    
    # Initial ATR using simple mean
    atr[period - 1] = np.mean(tr[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # EMA trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # ATR ratio for volatility regime (vol expansion = better mean reversion)
    atr_ratio = atr_7 / atr_30
    atr_ratio[np.isnan(atr_ratio)] = 1.0
    atr_ratio[np.isinf(atr_ratio)] = 1.0
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volatility regime: expansion = better mean reversion
        vol_expansion = atr_ratio[i] > 1.2
        
        # RSI signals (relaxed thresholds for trade frequency)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + price near/below lower BB + 4h bull trend
        if rsi_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        # Strong: RSI extreme oversold + 4h bull trend (ignore BB)
        elif rsi_extreme_oversold and bull_trend:
            new_signal = SIZE_MAX
        # Vol expansion: RSI oversold + vol expansion + 4h bull trend
        elif rsi_oversold and vol_expansion and bull_trend:
            new_signal = SIZE_BASE
        # EMA confirmation: RSI oversold + price below BB + EMA bullish
        elif rsi_oversold and price_below_bb_lower and ema_bullish:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + price near/above upper BB + 4h bear trend
        if rsi_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        # Strong: RSI extreme overbought + 4h bear trend (ignore BB)
        elif rsi_extreme_overbought and bear_trend:
            new_signal = -SIZE_MAX
        # Vol expansion: RSI overbought + vol expansion + 4h bear trend
        elif rsi_overbought and vol_expansion and bear_trend:
            new_signal = -SIZE_BASE
        # EMA confirmation: RSI overbought + price above BB + EMA bearish
        elif rsi_overbought and price_above_bb_upper and ema_bearish:
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