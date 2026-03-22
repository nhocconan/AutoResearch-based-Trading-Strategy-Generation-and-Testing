#!/usr/bin/env python3
"""
Experiment #168: 1d KAMA Adaptive Trend + 1w HMA Bias + BB Squeeze + RSI Entry

Hypothesis: 1d timeframe needs adaptive trend following that works in both bull and bear
markets. KAMA (Kaufman Adaptive Moving Average) adjusts sensitivity based on market
efficiency - fast in trends, slow in chop. Combined with 1w HMA for major trend bias,
BB squeeze for regime detection, and RSI for entry timing.

Why this might work on 1d:
- KAMA adapts to market conditions (unlike fixed EMA that failed in #158, #159, #160)
- 1w HTF provides stable major trend bias (avoid counter-trend trades)
- BB squeeze identifies low-volatility periods before breakouts
- RSI(7) extremes provide entry timing within the trend
- Fewer but higher-quality trades suitable for 1d timeframe

Learning from failures:
- #158, #159, #160: Simple EMA crossover failed on 30m, 1h, 4h
- #161: Donchian breakout on 12h had negative Sharpe
- #162: 1d HMA + 1w bias + MACD + BB had Sharpe=0.218 but discarded (likely <10 trades)
- #166: CRSI mean reversion on 4h failed catastrophically (Sharpe=-48)
- Need trend-following with adaptive parameters, not fixed EMA

Key differences from #162:
- KAMA instead of HMA for primary trend (more adaptive)
- RSI(7) instead of MACD for entry timing (faster signals)
- BB squeeze threshold tuned for 1d (lower threshold = more trades)
- Ensure entry conditions trigger frequently enough for ≥10 trades

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_1w_hma_bb_squeeze_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=7):
    """Calculate RSI with configurable period."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, min_periods=period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, period=7)
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (very stable)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === ADAPTIVE TREND (KAMA) ===
        # Price above KAMA = bullish momentum
        # Price below KAMA = bearish momentum
        bull_kama = close[i] > kama[i]
        bear_kama = close[i] < kama[i]
        
        # === BOLLINGER BAND SQUEEZE REGIME ===
        # Low bandwidth = squeeze (potential breakout coming)
        # Calculate bandwidth percentile approximation
        bb_squeeze = bb_bandwidth[i] < 0.10  # Low vol regime
        
        # === RSI ENTRY TIMING ===
        # In uptrend: enter on RSI pullback to 40-50
        # In downtrend: enter on RSI rally to 50-60
        rsi_long_entry = rsi[i] < 50 and rsi[i] > 35  # Pullback in uptrend
        rsi_short_entry = rsi[i] > 50 and rsi[i] < 65  # Rally in downtrend
        
        # === BB BOUNCE ENTRY ===
        # Long: price near lower band in uptrend
        # Short: price near upper band in downtrend
        price_near_lower = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10) < 0.3
        price_near_upper = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10) > 0.7
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 1w bullish + KAMA bullish + (RSI pullback OR BB bounce)
        long_condition_1 = bull_trend_1w and bull_kama and rsi_long_entry
        long_condition_2 = bull_trend_1w and bull_kama and price_near_lower
        
        if long_condition_1 or long_condition_2:
            # Stronger signal if BB squeeze present
            if bb_squeeze:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # 1w bearish + KAMA bearish + (RSI rally OR BB bounce)
        short_condition_1 = bear_trend_1w and bear_kama and rsi_short_entry
        short_condition_2 = bear_trend_1w and bear_kama and price_near_upper
        
        if short_condition_1 or short_condition_2:
            # Stronger signal if BB squeeze present
            if bb_squeeze:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals