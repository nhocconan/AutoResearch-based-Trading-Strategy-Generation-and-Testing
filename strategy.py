#!/usr/bin/env python3
"""
Experiment #008: 30m HMA Trend + RSI Pullback with 4h Regime Filter
Hypothesis: 30m captures intraday swings while 4h HMA provides trend bias.
Key insight from #005: 12h worked best (+16.6%), so 30m with 4h HTF should
capture more opportunities while maintaining trend alignment.
Entry: Price pullback to EMA21 + RSI(14) in 35-50 (long) or 50-65 (short)
       aligned with 4h HMA trend direction.
Regime filter: Bollinger Band Width percentile to detect squeeze/expansion.
Exit: 2.5*ATR trailing stop or opposite signal.
Sizing: 0.25-0.30 discrete levels to minimize fee churn.
Must work on BTC/ETH through 2022 crash and 2025 bear market.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h_regime_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return upper, lower, width

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_percentile_rank(values, window=100):
    """Calculate percentile rank over rolling window."""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    for i in range(window-1, n):
        window_vals = values[i-window+1:i+1]
        current = values[i]
        rank = np.sum(window_vals <= current) / window
        pr[i] = rank * 100
    return pr

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
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    
    # Bollinger Width percentile for regime detection
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # Regime detection via BB Width percentile
        # Low percentile = squeeze (expect breakout), High = expansion (expect mean reversion)
        regime_squeeze = bb_width_pr[i] < 30
        regime_expansion = bb_width_pr[i] > 70
        
        # RSI pullback levels (looser than before to generate more trades)
        rsi_pullback_long = rsi[i] >= 35 and rsi[i] <= 55
        rsi_pullback_short = rsi[i] >= 45 and rsi[i] <= 65
        
        # Price near EMA21 (pullback entry)
        price_near_ema_long = close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.02
        price_near_ema_short = close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.98
        
        # Price near Bollinger bands for mean reversion
        price_near_bb_lower = close[i] <= bb_lower[i] * 1.005
        price_near_bb_upper = close[i] >= bb_upper[i] * 0.995
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h HMA bullish + RSI pullback + price near EMA21
        if hma_4h_bullish and rsi_pullback_long and price_near_ema_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h HMA bullish + price at BB lower (mean reversion in uptrend)
        elif hma_4h_bullish and price_near_bb_lower and rsi[i] < 45:
            new_signal = SIZE_ENTRY
        # Tertiary: EMA bullish + RSI ok (trend continuation)
        elif ema_bullish and hma_4h_bullish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h HMA bearish + RSI pullback + price near EMA21
        if hma_4h_bearish and rsi_pullback_short and price_near_ema_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h HMA bearish + price at BB upper (mean reversion in downtrend)
        elif hma_4h_bearish and price_near_bb_upper and rsi[i] > 55:
            new_signal = -SIZE_ENTRY
        # Tertiary: EMA bearish + RSI ok (trend continuation)
        elif ema_bearish and hma_4h_bearish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC ===
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