#!/usr/bin/env python3
"""
Experiment #024: 1d Regime-Adaptive Hybrid (Mean Reversion + Trend Follow)
Hypothesis: Daily timeframe needs LOOSE entry conditions to generate sufficient trades.
Combines 1w HMA trend bias with 1d RSI/Z-score mean reversion entries.
Key insight from failures: Multiple conflicting filters = 0 trades. Use OR logic for entries.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Stoploss: 2.5*ATR trailing stop (wider for daily timeframe volatility).
Entry logic: ANY of (RSI extreme, Z-score extreme, BB touch) + trend filter = trade
This ensures ≥10 trades/symbol on train, ≥3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_hybrid_1w_hma_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using vectorized pandas."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    zscore = zscore.fillna(0.0)
    return zscore.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, bandwidth, sma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = np.nan_to_num(atr, nan=0.0)
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.fillna(0.0)
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    zscore = calculate_zscore(close, 20)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    
    # Additional trend filters
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Need 200 for EMA200 + 50 for warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF)
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # ADX regime
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        # RSI signals (LOOSE thresholds to ensure trades)
        rsi_oversold = rsi_14[i] < 40  # Was 30, too strict
        rsi_overbought = rsi_14[i] > 60  # Was 70, too strict
        rsi_extreme_oversold = rsi_7[i] < 30
        rsi_extreme_overbought = rsi_7[i] > 70
        
        # Z-score signals
        zscore_oversold = zscore[i] < -1.0  # Was -1.5, too strict
        zscore_overbought = zscore[i] > 1.0  # Was +1.5, too strict
        
        # Bollinger Band signals
        price_at_lower = close[i] <= bb_lower[i] * 1.01
        price_at_upper = close[i] >= bb_upper[i] * 0.99
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_20[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_20[i] and close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (ANY of these conditions = trade) ===
        # Condition 1: RSI oversold + bull trend
        if rsi_oversold and bull_trend:
            new_signal = SIZE_BASE
        
        # Condition 2: Z-score oversold + bull trend
        elif zscore_oversold and bull_trend:
            new_signal = SIZE_BASE
        
        # Condition 3: Price at lower BB + bull trend
        elif price_at_lower and bull_trend:
            new_signal = SIZE_BASE
        
        # Condition 4: Extreme RSI (any trend)
        elif rsi_extreme_oversold:
            new_signal = SIZE_BASE
        
        # Condition 5: Strong confluence (RSI + Z-score + trend)
        elif rsi_oversold and zscore_oversold and bull_trend:
            new_signal = SIZE_MAX
        
        # Condition 6: Ranging market + mean reversion
        elif ranging and rsi_oversold and price_at_lower:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY (ANY of these conditions = trade) ===
        # Condition 1: RSI overbought + bear trend
        if rsi_overbought and bear_trend:
            new_signal = -SIZE_BASE
        
        # Condition 2: Z-score overbought + bear trend
        elif zscore_overbought and bear_trend:
            new_signal = -SIZE_BASE
        
        # Condition 3: Price at upper BB + bear trend
        elif price_at_upper and bear_trend:
            new_signal = -SIZE_BASE
        
        # Condition 4: Extreme RSI (any trend)
        elif rsi_extreme_overbought:
            new_signal = -SIZE_BASE
        
        # Condition 5: Strong confluence (RSI + Z-score + trend)
        elif rsi_overbought and zscore_overbought and bear_trend:
            new_signal = -SIZE_MAX
        
        # Condition 6: Ranging market + mean reversion
        elif ranging and rsi_overbought and price_at_upper:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily)
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
            
            # Calculate trailing stop (2.5*ATR for daily)
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