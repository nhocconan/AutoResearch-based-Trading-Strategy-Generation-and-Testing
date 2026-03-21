#!/usr/bin/env python3
"""
Hypothesis: 30m primary with 4h HMA trend filter + RSI pullback entries + regime detection.
The 4h trend provides directional bias, 30m RSI finds pullback entries in trend direction.
Bollinger BandWidth percentile detects regime (trend vs range) to adjust entry thresholds.
ATR-based stoploss protects against large drawdowns. Discrete sizing minimizes fee churn.
Timeframe: 30m (required for experiment #002)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_regime_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bollinger_bandwidth(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return bandwidth, sma, std

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop ( Rule 1 )
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    hma_30m = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    bb_width, bb_sma, bb_std = calculate_bollinger_bandwidth(close, 20, 2.0)
    
    # Calculate BB width percentile for regime detection (rolling 100 bars)
    bb_width_series = pd.Series(bb_width)
    bb_percentile = bb_width_series.rolling(window=100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x.values), x.iloc[-1]) / len(x), raw=False
    ).values
    bb_percentile = np.nan_to_num(bb_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0
    
    for i in range(100, n):
        # HTF trend from 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h_val
        htf_trend = 1.0 if hma_4h_val > hma_4h_prev else -1.0
        
        # LTF trend from 30m HMA
        ltf_trend = 1.0 if hma_30m[i] > hma_30m[i-1] else -1.0
        
        # Regime: low BB width = range, high BB width = trend
        is_trend_regime = bb_percentile[i] > 0.6
        is_range_regime = bb_percentile[i] < 0.4
        
        # RSI levels adjusted by regime
        if is_trend_regime:
            rsi_long_threshold = 45
            rsi_short_threshold = 55
        elif is_range_regime:
            rsi_long_threshold = 35
            rsi_short_threshold = 65
        else:
            rsi_long_threshold = 40
            rsi_short_threshold = 60
        
        current_signal = 0.0
        
        # Long entry: HTF bullish + LTF pullback + RSI not overbought
        if htf_trend > 0 and ltf_trend > 0:
            if rsi[i] < rsi_long_threshold and rsi[i] > 25:
                current_signal = SIZE_LONG
                entry_price = close[i]
                position_side = 1
        
        # Short entry: HTF bearish + LTF pullback + RSI not oversold
        elif htf_trend < 0 and ltf_trend < 0:
            if rsi[i] > rsi_short_threshold and rsi[i] < 75:
                current_signal = -SIZE_SHORT
                entry_price = close[i]
                position_side = -1
        
        # Stoploss logic (Rule 6)
        if position_side == 1 and prev_signal > 0:
            if close[i] < entry_price - 2.5 * atr[i]:
                current_signal = 0.0
                position_side = 0
            elif rsi[i] > 70:  # Take profit on overbought
                current_signal = SIZE_LONG / 2
        
        if position_side == -1 and prev_signal < 0:
            if close[i] > entry_price + 2.5 * atr[i]:
                current_signal = 0.0
                position_side = 0
            elif rsi[i] < 30:  # Take profit on oversold
                current_signal = -SIZE_SHORT / 2
        
        # Smooth signal changes to reduce churn
        if abs(current_signal - prev_signal) > 0.01:
            signals[i] = current_signal
        else:
            signals[i] = prev_signal
        
        prev_signal = signals[i]
    
    return signals