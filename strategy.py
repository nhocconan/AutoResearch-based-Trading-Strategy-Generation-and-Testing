#!/usr/bin/env python3
"""
Experiment #019: 15m HMA Trend + RSI/Stoch Pullback with Choppiness Filter
Hypothesis: 15m strategies failed before due to excessive trading and no regime filter.
This uses 4h HMA for trend bias (proven in baseline), 15m RSI+Stoch for entries,
Choppiness Index to avoid whipsaw in ranging markets, and ATR stops.
Key difference: Fewer, higher-quality trades by requiring multiple confirmations.
Position sizing: 0.25 base, 0.35 in aligned HTF trend, stoploss at 2.5*ATR.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_stoch_chop_4h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    
    rs = np.zeros(n)
    rs[period:] = np.where(avg_loss[period:] != 0, avg_gain[period:] / avg_loss[period:], 100.0)
    rsi[period:] = 100.0 - (100.0 / (1.0 + rs[period:]))
    
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator %K and %D."""
    n = len(close)
    k = np.zeros(n)
    k[:] = np.nan
    d = np.zeros(n)
    d[:] = np.nan
    
    for i in range(k_period - 1, n):
        ll = np.min(low[i - k_period + 1:i + 1])
        hh = np.max(high[i - k_period + 1:i + 1])
        if hh > ll:
            k[i] = 100.0 * (close[i] - ll) / (hh - ll)
        else:
            k[i] = 50.0
    
    k_series = pd.Series(k)
    d_vals = k_series.rolling(window=d_period, min_periods=d_period).mean().values
    d[:] = d_vals
    
    return k, d

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - identifies ranging vs trending markets.
    CHOP > 61.8 = ranging (mean reversion favorable)
    CHOP < 38.2 = trending (trend following favorable)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            tr_sum += tr
        
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if tr_sum > 0 and hh > ll:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2.0 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    rsi = calculate_rsi(close, 14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    chop = calculate_choppiness(high, low, close, 14)
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Additional trend filters
    ema_55 = pd.Series(close).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_TREND = 0.35
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = SIZE_EXIT
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = SIZE_EXIT
            continue
        
        if np.isnan(rsi[i]) or np.isnan(stoch_k[i]) or np.isnan(chop[i]):
            signals[i] = SIZE_EXIT
            continue
        
        if np.isnan(kama[i]) or np.isnan(bb_upper[i]):
            signals[i] = SIZE_EXIT
            continue
        
        # 4h regime bias (HTF) - determines which direction to favor
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # KAMA trend direction (slope over 5 bars)
        kama_slope = 0.0
        if i > 5 and not np.isnan(kama[i - 5]):
            kama_slope = (kama[i] - kama[i - 5]) / kama[i - 5] if kama[i - 5] != 0 else 0.0
        kama_rising = kama_slope > 0.0001
        kama_falling = kama_slope < -0.0001
        
        # Price position vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_55[i] and (np.isnan(ema_200[i]) or close[i] > ema_200[i])
        ema_bearish = close[i] < ema_55[i] and (np.isnan(ema_200[i]) or close[i] < ema_200[i])
        
        # RSI signals
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral = 40.0 < rsi[i] < 60.0
        
        # Stochastic signals
        stoch_oversold = stoch_k[i] < 25.0 and stoch_d[i] < 25.0
        stoch_overbought = stoch_k[i] > 75.0 and stoch_d[i] > 75.0
        stoch_bull_cross = stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1] if i > 0 else False
        stoch_bear_cross = stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1] if i > 0 else False
        
        # Bollinger Band signals
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] != 0 else 0.0
        at_lower_band = close[i] <= bb_lower[i] * 1.005
        at_upper_band = close[i] >= bb_upper[i] * 0.995
        
        # Choppiness filter
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # Select position size based on regime alignment
        current_size = SIZE_TREND if bull_regime else SIZE_BASE
        
        new_signal = SIZE_EXIT
        
        # === LONG ENTRY ===
        # Primary: Bull regime + RSI oversold + price above KAMA + trending
        if bull_regime and rsi_oversold and price_above_kama and is_trending:
            new_signal = current_size
        # Secondary: Bull regime + Stoch cross from oversold + EMA bullish
        elif bull_regime and stoch_oversold and stoch_bull_cross and ema_bullish:
            new_signal = current_size
        # Tertiary: Bull regime + BB lower band + RSI < 50 (mean reversion in trend)
        elif bull_regime and at_lower_band and rsi[i] < 50.0:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Bear regime + RSI overbought + price below KAMA + trending
        if bear_regime and rsi_overbought and price_below_kama and is_trending:
            new_signal = -current_size
        # Secondary: Bear regime + Stoch cross from overbought + EMA bearish
        elif bear_regime and stoch_overbought and stoch_bear_cross and ema_bearish:
            new_signal = -current_size
        # Tertiary: Bear regime + BB upper band + RSI > 50 (mean reversion in trend)
        elif bear_regime and at_upper_band and rsi[i] > 50.0:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
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
                new_signal = SIZE_EXIT
        
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
                new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else SIZE_EXIT
        
        # New position opened
        if new_signal != SIZE_EXIT and prev_signal == SIZE_EXIT:
            entry_price = close[i]
            position_side = 1 if new_signal > 0 else -1
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != SIZE_EXIT and prev_signal != SIZE_EXIT and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = 1 if new_signal > 0 else -1
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == SIZE_EXIT and prev_signal != SIZE_EXIT:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals