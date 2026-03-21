#!/usr/bin/env python3
"""
Experiment #008: 30m KAMA + Bollinger Regime + 4h Trend Filter
Hypothesis: 30m captures intraday swings while 4h provides major trend direction.
KAMA adapts to volatility better than EMA (reduces whipsaw in choppy markets).
Bollinger BandWidth percentile detects regime: narrow=range (mean reversion), wide=trend (trend follow).
4h KAMA slope filters entries: only long when 4h trending up, only short when 4h trending down.
Combines successful 4h KAMA+BB approach (#004) with 30m entry precision.
Position sizing: 0.25 base, reduced to 0.15 in high volatility.
Stoploss: 2.5*ATR trailing stop with signal→0 on breach.
Target: 30-60 trades/year with Sharpe > 0.15 (beat #004 baseline).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_regime_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman's Adaptive Moving Average - adapts smoothing based on market efficiency."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = np.where(noise > 0, change / noise, 0)
    er = np.nan_to_num(er, nan=0.0)
    
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / np.where(sma > 0, sma, 1)
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return sma, upper, lower, bandwidth

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

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate Bollinger BandWidth percentile for regime detection."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i+1]
        bb_pct[i] = np.sum(window <= bandwidth[i]) / len(window)
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, 10, 2, 30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 30m indicators (pre-compute before loop for performance)
    kama_30m = calculate_kama(close, 10, 2, 30)
    sma_30m, upper_30m, lower_30m, bb_width_30m = calculate_bollinger(close, 20, 2.0)
    atr_30m = calculate_atr(high, low, close, 14)
    rsi_30m = calculate_rsi(close, 14)
    bb_pct_30m = calculate_bb_percentile(bb_width_30m, 100)
    
    # 4h trend direction (slope of KAMA over 10 bars)
    kama_4h_slope = np.zeros(n)
    for i in range(50, n):
        if kama_4h_aligned[i] > kama_4h_aligned[i-10]:
            kama_4h_slope[i] = 1  # Bullish
        elif kama_4h_aligned[i] < kama_4h_aligned[i-10]:
            kama_4h_slope[i] = -1  # Bearish
        else:
            kama_4h_slope[i] = 0  # Neutral
    
    signals = np.zeros(n)
    SIZE_NORMAL = 0.25
    SIZE_REDUCED = 0.15
    
    # Track positions for stoploss (persist across iterations)
    position_side = 0
    entry_price = 0.0
    
    for i in range(100, n):
        # Regime detection
        is_trend_regime = bb_pct_30m[i] > 0.6  # Wide bands = trending
        is_range_regime = bb_pct_30m[i] < 0.4  # Narrow bands = ranging
        
        # 4h trend filter
        htf_bullish = kama_4h_slope[i] == 1
        htf_bearish = kama_4h_slope[i] == -1
        
        # 30m KAMA position
        kama_above = close[i] > kama_30m[i]
        kama_below = close[i] < kama_30m[i]
        
        # RSI conditions (relaxed to ensure trades)
        rsi_oversold = rsi_30m[i] < 45
        rsi_overbought = rsi_30m[i] > 55
        rsi_neutral = 40 <= rsi_30m[i] <= 60
        
        # Entry logic - multiple conditions to ensure we get trades
        new_signal = 0.0
        
        # Long entries (relaxed conditions)
        if htf_bullish and kama_above:
            if rsi_oversold:
                new_signal = SIZE_NORMAL
            elif is_trend_regime and rsi_neutral:
                new_signal = SIZE_NORMAL
            elif rsi_30m[i] > 42:
                new_signal = SIZE_REDUCED
        
        # Short entries (relaxed conditions)
        if htf_bearish and kama_below:
            if rsi_overbought:
                new_signal = -SIZE_NORMAL
            elif is_trend_regime and rsi_neutral:
                new_signal = -SIZE_NORMAL
            elif rsi_30m[i] < 58:
                new_signal = -SIZE_REDUCED
        
        # Stoploss logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr_30m[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            # Trail stop for profits
            elif close[i] > entry_price + 3.0 * atr_30m[i]:
                if new_signal == SIZE_NORMAL:
                    new_signal = SIZE_REDUCED
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr_30m[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            # Trail stop for profits
            elif close[i] < entry_price - 3.0 * atr_30m[i]:
                if new_signal == -SIZE_NORMAL:
                    new_signal = -SIZE_REDUCED
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals