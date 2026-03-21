#!/usr/bin/env python3
"""
Hypothesis: 15m primary with 4h HMA trend filter + 1h RSI entry timing + ATR stoploss.
Multi-timeframe approach: 4h determines bias, 1h RSI finds pullback entries, 15m ATR manages risk.
Regime filter via Bollinger BandWidth to avoid choppy markets. Discrete sizing (0.0, ±0.25, ±0.35).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_atr_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average."""
    s = pd.Series(series)
    wma1 = s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hull.values

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
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
    """Calculate Bollinger Band Width as regime filter."""
    s = pd.Series(close)
    sma = s.rolling(window=period, min_periods=period).mean().values
    std = s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma
    return bw

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_prev = np.roll(hma_4h, 1)
    hma_4h_prev[0] = hma_4h[0]
    trend_4h = np.where(hma_4h > hma_4h_prev, 1.0, -1.0)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1h RSI for entry timing
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, 21)
    hma_15m_prev = np.roll(hma_15m, 1)
    hma_15m_prev[0] = hma_15m[0]
    
    rsi_15m = calculate_rsi(close, 14)
    atr_15m = calculate_atr(high, low, close, 14)
    bbw_15m = calculate_bollinger_bandwidth(close, 20, 2.0)
    
    # Calculate BBW percentile for regime detection
    bbw_percentile = pd.Series(bbw_15m).rolling(window=100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x.dropna()), x.iloc[-1]) / len(x.dropna()) if len(x.dropna()) > 0 else 0.5,
        raw=False
    ).values
    bbw_percentile = np.nan_to_num(bbw_percentile, nan=0.5)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0
    
    for i in range(100, n):
        # Regime filter: avoid very low volatility (chop) and very high (chaos)
        regime_ok = (bbw_percentile[i] > 0.2) and (bbw_percentile[i] < 0.85)
        
        # 4h trend bias
        trend_bias = trend_4h_aligned[i]
        
        # 1h RSI for entry timing
        rsi_1h_val = rsi_1h_aligned[i]
        rsi_15m_val = rsi_15m[i]
        
        # Entry conditions
        long_signal = False
        short_signal = False
        
        if regime_ok:
            # Long: 4h uptrend + 1h RSI not overbought + 15m RSI pullback
            if trend_bias > 0:
                if rsi_1h_val < 65 and rsi_15m_val < 60 and rsi_15m_val > 35:
                    long_signal = True
            
            # Short: 4h downtrend + 1h RSI not oversold + 15m RSI bounce
            if trend_bias < 0:
                if rsi_1h_val > 35 and rsi_15m_val > 40 and rsi_15m_val < 65:
                    short_signal = True
        
        # Stoploss logic (Rule 6)
        if position_side != 0 and entry_price > 0:
            if position_side > 0 and close[i] < entry_price - 2.5 * atr_15m[i]:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
            if position_side < 0 and close[i] > entry_price + 2.5 * atr_15m[i]:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Generate signal
        if long_signal and position_side <= 0:
            signals[i] = SIZE_LONG
            entry_price = close[i]
            position_side = 1
        elif short_signal and position_side >= 0:
            signals[i] = -SIZE_SHORT
            entry_price = close[i]
            position_side = -1
        elif not long_signal and not short_signal:
            # Hold position or go flat
            if position_side == 1:
                signals[i] = SIZE_LONG
            elif position_side == -1:
                signals[i] = -SIZE_SHORT
            else:
                signals[i] = 0.0
        else:
            signals[i] = prev_signal
        
        prev_signal = signals[i]
    
    return signals