#!/usr/bin/env python3
"""
Experiment #031: 15m RSI + Z-score Mean Reversion with 4h HMA Trend Filter
Hypothesis: 15m timeframe offers more mean reversion opportunities than 1h/4h.
Combining RSI(14) extremes with Z-score(20) filter captures oversold/overbought conditions.
4h HMA trend filter prevents counter-trend trades in strong trends.
Bollinger Band regime detection adjusts entry thresholds based on volatility.
ATR trailing stop at 2.0*ATR limits drawdown on adverse moves.
Timeframe: 15m (REQUIRED for this experiment), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Less restrictive entry thresholds (RSI 35/65, Z-score ±1.5) ensure sufficient trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_zscore_4h_hma_meanrev_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.fillna(0.0).values
    return zscore

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
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
    return atr

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, period=14)
    zscore = calculate_zscore(close, period=20)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filter on 15m
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Bollinger bandwidth percentile for regime detection
    bb_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / len(x[:-1]) * 100 if len(x) > 1 else 50
    ).values
    bb_percentile = np.nan_to_num(bb_percentile, nan=50.0)
    
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
    
    for i in range(250, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary trend filter
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # 15m EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        # Bollinger regime: narrow bandwidth = range, wide = trending
        range_regime = bb_percentile[i] < 40  # Bottom 40% = ranging
        trend_regime = bb_percentile[i] > 60  # Top 40% = trending
        
        # RSI signals (mean reversion) - LESS RESTRICTIVE for more trades
        rsi_oversold = rsi[i] < 40  # Not too extreme
        rsi_overbought = rsi[i] > 60  # Not too extreme
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        
        # Z-score signals
        zscore_oversold = zscore[i] < -1.0  # Less restrictive
        zscore_overbought = zscore[i] > 1.0  # Less restrictive
        zscore_extreme_oversold = zscore[i] < -1.5
        zscore_extreme_overbought = zscore[i] > 1.5
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + Z-score oversold + 4h bull trend
        if rsi_oversold and zscore_oversold and bull_trend:
            new_signal = SIZE_BASE
        
        # Secondary: RSI extreme oversold + price near lower BB (any trend)
        if rsi_extreme_oversold and price_near_lower:
            new_signal = max(new_signal, SIZE_BASE)
        
        # Tertiary: RSI oversold + range regime + price above 4h HMA
        if rsi_oversold and range_regime and bull_trend:
            new_signal = max(new_signal, SIZE_BASE)
        
        # Strong long: Multiple confirmations
        if rsi_extreme_oversold and zscore_extreme_oversold and bull_trend:
            new_signal = SIZE_MAX
        
        # Strong long: Price breaks below BB + oversold conditions
        if price_below_lower and rsi_oversold and zscore_oversold:
            new_signal = max(new_signal, SIZE_MAX)
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + Z-score overbought + 4h bear trend
        if rsi_overbought and zscore_overbought and bear_trend:
            new_signal = -SIZE_BASE
        
        # Secondary: RSI extreme overbought + price near upper BB (any trend)
        if rsi_extreme_overbought and price_near_upper:
            new_signal = min(new_signal, -SIZE_BASE)
        
        # Tertiary: RSI overbought + range regime + price below 4h HMA
        if rsi_overbought and range_regime and bear_trend:
            new_signal = min(new_signal, -SIZE_BASE)
        
        # Strong short: Multiple confirmations
        if rsi_extreme_overbought and zscore_extreme_overbought and bear_trend:
            new_signal = -SIZE_MAX
        
        # Strong short: Price breaks above BB + overbought conditions
        if price_above_upper and rsi_overbought and zscore_overbought:
            new_signal = min(new_signal, -SIZE_MAX)
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr[i]
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
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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