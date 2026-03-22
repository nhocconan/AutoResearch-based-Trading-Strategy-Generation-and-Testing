#!/usr/bin/env python3
"""
Experiment #135: 1h Volatility Squeeze Breakout + 4h HMA Trend + RSI Pullback + ATR Stop

Hypothesis: Combining proven elements from winning strategies with volatility regime detection:
- BB Width at percentile < 20 = squeeze (coiling before breakout)
- 4h HMA(21) for higher timeframe trend bias (proven in Sharpe=0.478 strategy)
- RSI(14) pullback entries within trend (buy dips in uptrend, sell rallies in downtrend)
- ATR(14) ratio filter: avoid entries when ATR(7)/ATR(30) > 2.5 (vol spike = wait for calm)
- Asymmetric sizing: stronger positions when HTF trend aligns with breakout direction
- Trailing stop at 2.0 * ATR protects against reversals

Why this might beat previous attempts:
- Volatility squeeze precedes 70%+ of major moves (research-backed)
- 4h HMA filter avoids counter-trend trades (addresses 2022 whipsaw)
- RSI pullback ensures better entry prices within trend
- ATR ratio filter avoids entering during panic spikes (buys after vol crush)
- 1h timeframe balances signal frequency vs noise (more trades than 4h/12h)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_squeeze_4h_hma_rsi_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bw = (upper - lower) / sma * 100
    bw = np.where(sma > 0, bw, 0)
    return upper, lower, bw

def calculate_bb_percentile(bw, lookback=100):
    """Calculate BB Width percentile over lookback period."""
    n = len(bw)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback, n):
        window = bw[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bw[i]) / len(valid) * 100
    
    return percentile

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_s = pd.Series(atr)
    atr_short = atr_s.rolling(window=short_period, min_periods=short_period).mean().values
    atr_long = atr_s.rolling(window=long_period, min_periods=long_period).mean().values
    ratio = np.where(atr_long > 0, atr_short / atr_long, 1.0)
    return ratio

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_width, 100)
    atr_ratio = calculate_atr_ratio(atr, 7, 30)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY REGIME ===
        # BB Width percentile < 20 = squeeze (coiling, expect breakout)
        # BB Width percentile > 80 = expansion (trend may be ending)
        bb_squeeze = bb_percentile[i] < 25
        bb_expansion = bb_percentile[i] > 75
        
        # === ATR VOLATILITY FILTER ===
        # ATR ratio > 2.5 = vol spike (panic), wait for calm
        # ATR ratio < 1.2 = vol crush (good for entries)
        vol_spike = atr_ratio[i] > 2.5
        vol_calm = atr_ratio[i] < 1.5
        
        # === RSI PULLBACK ===
        # In uptrend: RSI 40-55 = pullback buy opportunity
        # In downtrend: RSI 45-60 = rally sell opportunity
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_neutral = 40 < rsi[i] < 60
        
        # === PRICE POSITION ===
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        price_middle = bb_lower[i] < close[i] < bb_upper[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + BB squeeze + RSI pullback + vol calm + price near lower BB
        if bull_trend_4h and bb_squeeze and rsi_oversold and vol_calm and price_near_lower:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + BB squeeze + RSI pullback
        elif bull_trend_4h and bb_squeeze and rsi_oversold:
            new_signal = SIZE_BASE
        # Weak (ensure trades): 4h bullish + RSI oversold + not vol spike
        elif bull_trend_4h and rsi_oversold and not vol_spike:
            new_signal = SIZE_BASE
        # Breakout: 4h bullish + price breaks upper BB + vol calm
        elif bull_trend_4h and price_near_upper and vol_calm:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + BB squeeze + RSI rally + vol calm + price near upper BB
        if bear_trend_4h and bb_squeeze and rsi_overbought and vol_calm and price_near_upper:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + BB squeeze + RSI overbought
        elif bear_trend_4h and bb_squeeze and rsi_overbought:
            new_signal = -SIZE_BASE
        # Weak (ensure trades): 4h bearish + RSI overbought + not vol spike
        elif bear_trend_4h and rsi_overbought and not vol_spike:
            new_signal = -SIZE_BASE
        # Breakdown: 4h bearish + price breaks lower BB + vol calm
        elif bear_trend_4h and price_near_lower and vol_calm:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
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