#!/usr/bin/env python3
"""
Experiment #164: 30m BB-KC Squeeze + 4h HMA Trend + ADX Confirmation + ATR Stop

Hypothesis: Bollinger Band / Keltner Channel squeeze identifies low-volatility 
compression periods that precede explosive moves. Combined with 4h HMA trend 
filter and ADX confirmation, this should capture high-probability breakouts 
on 30m timeframe. This is DIFFERENT from failed RSI/EMA strategies (#152, #157, #163).

Why this might work on 30m:
- BB-KC squeeze is a proven volatility contraction pattern (John Carter's TTM Squeeze)
- 30m captures intraday moves without excessive noise of 5m/15m
- 4h HMA provides stable trend bias (proven in current best strategy)
- ADX > 20 filters choppy periods where squeezes fail to breakout
- ATR-based stoploss protects against false breakouts

Learning from failures:
- #152, #157, #163: RSI-based strategies failed on 30m/15m
- #158: Simple EMA crossover failed on 30m
- Need volatility-based entry, not momentum-based
- Squeeze patterns work in both bull and bear markets

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bbkc_squeeze_4h_hma_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100  # Normalized bandwidth
    return upper, lower, sma, bandwidth

def calculate_keltner_channels(high, low, close, period=20, atr_mult=1.5, atr_period=14):
    """Calculate Keltner Channels."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, lower, ema

def calculate_bb_percentile(bandwidth, lookback=30):
    """Calculate bandwidth percentile over lookback period."""
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback-1, n):
        window = bandwidth[i-lookback+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid < bandwidth[i]) / len(valid) * 100
        else:
            percentile[i] = 50.0
    
    return percentile

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
    adx = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, 30)
    
    # Keltner Channels
    kc_upper, kc_lower, kc_mid = calculate_keltner_channels(high, low, close, 20, 1.5, 14)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(bb_bandwidth[i]) or np.isnan(kc_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # BB inside KC = squeeze (low volatility compression)
        squeeze_on = (bb_upper[i] < kc_upper[i]) and (bb_lower[i] > kc_lower[i])
        
        # Bandwidth at low percentile = extreme compression
        squeeze_extreme = bb_percentile[i] < 20  # Bottom 20% of bandwidth
        
        # === TREND STRENGTH FILTER ===
        # ADX > 18 = trending market (breakouts more likely to succeed)
        # Using 18 instead of 20 for more trades on 30m
        trend_strength = adx[i] > 18
        
        # === BREAKOUT SIGNAL ===
        # Price closes above BB upper = bullish breakout
        # Price closes below BB lower = bearish breakout
        breakout_long = close[i] > bb_upper[i-1]
        breakout_short = close[i] < bb_lower[i-1]
        
        # === MOMENTUM CONFIRMATION ===
        # Price above KC mid = bullish momentum
        # Price below KC mid = bearish momentum
        momentum_long = close[i] > kc_mid[i]
        momentum_short = close[i] < kc_mid[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + (squeeze + breakout OR extreme squeeze + momentum) + ADX
        if bull_trend_4h and trend_strength and momentum_long:
            if squeeze_on and breakout_long:
                new_signal = SIZE_BASE
            elif squeeze_extreme and breakout_long:
                new_signal = SIZE_STRONG
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + (squeeze + breakout OR extreme squeeze + momentum) + ADX
        if bear_trend_4h and trend_strength and momentum_short:
            if squeeze_on and breakout_short:
                new_signal = -SIZE_BASE
            elif squeeze_extreme and breakout_short:
                new_signal = -SIZE_STRONG
        
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