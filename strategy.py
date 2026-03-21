#!/usr/bin/env python3
"""
Experiment #007: 15m Supertrend + RSI Mean Reversion + 4h HMA Trend Filter
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides major trend direction.
Supertrend(10,3) gives clear trend signals with ATR-based stops built in.
RSI(14) extremes (30/70) provide mean reversion entries in direction of HTF trend.
Bollinger BandWidth percentile detects regime (squeeze=range, expansion=trend).
Position sizing: 0.25 discrete levels with 2.5*ATR stoploss.
This should work in both bull (2021) and bear/range (2025) markets by adapting to regime.
Key: Only trade when 4h trend and 15m signal align, reducing whipsaw in 2022 crash.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_4h_v1"
timeframe = "15m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and BandWidth."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, sma, bandwidth

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate Bollinger BandWidth percentile for regime detection."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i]
        bb_pct[i] = np.sum(window < bandwidth[i]) / lookback
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_percentile(bb_bandwidth, 100)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    
    for i in range(150, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Regime detection (BB percentile)
        regime_squeeze = bb_percentile[i] < 0.3  # Low bandwidth = range
        regime_expansion = bb_percentile[i] > 0.7  # High bandwidth = trend
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Entry logic - multiple conditions for quality signals
        new_signal = 0.0
        
        # Long entry: 4h bullish + Supertrend bullish + RSI not overbought
        if trend_4h_bullish and st_bullish and rsi[i] < 65:
            # Stronger signal if RSI pulls back from overbought
            if rsi[i] > 45 and rsi[i-1] < rsi[i]:
                new_signal = SIZE
            # Or RSI recovery from oversold
            elif rsi_oversold and rsi[i] > rsi[i-1]:
                new_signal = SIZE
        
        # Short entry: 4h bearish + Supertrend bearish + RSI not oversold
        elif trend_4h_bearish and st_bearish and rsi[i] > 35:
            # Stronger signal if RSI pulls back from oversold
            if rsi[i] < 55 and rsi[i-1] > rsi[i]:
                new_signal = -SIZE
            # Or RSI rejection from overbought
            elif rsi_overbought and rsi[i] < rsi[i-1]:
                new_signal = -SIZE
        
        # Range regime: mean reversion trades only
        if regime_squeeze and not regime_expansion:
            # In squeeze, trade RSI extremes against the move
            if rsi[i] < 30 and rsi[i] > rsi[i-1] and trend_4h_bullish:
                new_signal = SIZE * 0.8  # Smaller size in range
            elif rsi[i] > 70 and rsi[i] < rsi[i-1] and trend_4h_bearish:
                new_signal = -SIZE * 0.8
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] == SIZE:  # Only reduce if at full size
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] == -SIZE:  # Only reduce if at full size
                    new_signal = -HALF_SIZE
        
        # Supertrend reversal exit
        if position_side > 0 and st_bearish:
            new_signal = 0.0
        if position_side < 0 and st_bullish:
            new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_price[i] = close[i]
            lowest_price[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
            highest_price[i] = max(highest_price[i-1], close[i])
            lowest_price[i] = min(lowest_price[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_price[i] = highest_price[i-1] if i > 0 else close[i]
            lowest_price[i] = lowest_price[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals