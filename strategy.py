#!/usr/bin/env python3
"""
Experiment #018: 1d Regime-Adaptive Z-Score + Donchian Breakout + 4h HMA Trend
Hypothesis: Daily timeframe captures major regime changes better than lower TFs.
Combines mean reversion (Z-score) in range markets with breakout (Donchian) in trend markets.
4h HMA provides trend bias to avoid counter-trend trades. Bollinger bandwidth detects regime.
ATR trailing stop at 2.5*ATR limits drawdown on daily moves.
Timeframe: 1d (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.20-0.35 discrete levels, asymmetric based on regime confidence.
Key innovation: Regime-adaptive logic switches between mean reversion and breakout strategies.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_zscore_donchian_4h_hma_v1"
timeframe = "1d"
leverage = 1.0

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / std
    zscore[np.isnan(zscore)] = 0.0
    return zscore

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channels for breakout signals."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / atr[i]
            minus_di[i] = 100 * minus_dm_s[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Bollinger bandwidth percentile for regime detection
    bb_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=100).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / len(x[:-1]) * 100 if len(x) > 1 else 50
    ).values
    bb_percentile[np.isnan(bb_percentile)] = 50.0
    
    # ADX rolling average for trend confirmation
    adx_avg = pd.Series(adx).rolling(window=10, min_periods=10).mean().values
    adx_avg[np.isnan(adx_avg)] = 20.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Regime detection
        range_regime = bb_percentile[i] < 40  # Bottom 40% = ranging
        trend_regime = bb_percentile[i] > 60  # Top 40% = trending
        strong_trend = adx[i] > 25 and adx_avg[i] > 20
        
        # Z-score mean reversion signals
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_extreme_oversold = zscore[i] < -2.0
        zscore_extreme_overbought = zscore[i] > 2.0
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.005
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion: Z-score extreme oversold + range regime + 4h bull trend
        if zscore_extreme_oversold and range_regime and bull_trend:
            new_signal = SIZE_MAX
        # Mean reversion: Z-score oversold + price near lower BB + 4h bull trend
        elif zscore_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        # Breakout: Donchian breakout + strong trend + 4h bull trend
        elif donchian_breakout_long and strong_trend and bull_trend:
            new_signal = SIZE_MAX
        # Breakout: Donchian breakout + EMA bullish + 4h bull trend
        elif donchian_breakout_long and ema_bullish and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Mean reversion: Z-score extreme overbought + range regime + 4h bear trend
        if zscore_extreme_overbought and range_regime and bear_trend:
            new_signal = -SIZE_MAX
        # Mean reversion: Z-score overbought + price near upper BB + 4h bear trend
        elif zscore_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        # Breakout: Donchian breakout + strong trend + 4h bear trend
        elif donchian_breakout_short and strong_trend and bear_trend:
            new_signal = -SIZE_MAX
        # Breakout: Donchian breakout + EMA bearish + 4h bear trend
        elif donchian_breakout_short and ema_bearish and bear_trend:
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