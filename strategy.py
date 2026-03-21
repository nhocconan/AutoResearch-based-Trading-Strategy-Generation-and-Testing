#!/usr/bin/env python3
"""
Experiment #109: 15m RSI Mean Reversion with 4h HMA Trend Filter + BB Regime
Hypothesis: 15m timeframe is noisy for pure trend following. Instead, use HTF (4h) 
HMA for trend direction, then enter on 15m RSI extremes IN DIRECTION of HTF trend.
Add Bollinger Band width percentile for regime detection - wide bands = mean revert,
narrow bands = breakout. This combines proven 4h HMA filter with 15m mean reversion.
Position sizing: 0.25 entry, stoploss at 2*ATR, take profit at 2R. Target 30-50 trades/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_4h_hma_bb_regime_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, sma, bandwidth

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate bandwidth percentile over lookback period."""
    bb_pct = np.zeros(len(bandwidth))
    for i in range(lookback, len(bandwidth)):
        window = bandwidth[i-lookback:i+1]
        bb_pct[i] = np.sum(window <= bandwidth[i]) / len(window)
    return bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger(close, 20, 2.0)
    bb_pct = calculate_bb_percentile(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    take_profit_price = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        htftrend_bullish = close[i] > hma_4h_aligned[i]
        htftrend_bearish = close[i] < hma_4h_aligned[i]
        
        # Bollinger Band regime
        # bb_pct > 0.7 = wide bands (mean reversion likely)
        # bb_pct < 0.3 = narrow bands (breakout likely, avoid mean reversion)
        bb_wide = bb_pct[i] > 0.6
        bb_narrow = bb_pct[i] < 0.3
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 30) and (rsi[i] <= 70)
        
        # Price position relative to BB
        price_near_lower = close[i] <= bb_lower[i] * 1.002
        price_near_upper = close[i] >= bb_upper[i] * 0.998
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI oversold + price near lower BB + BB wide
        if htftrend_bullish and rsi_oversold and price_near_lower and bb_wide:
            new_signal = SIZE_ENTRY
        # LONG ENTRY 2: 4h bullish + RSI coming out of oversold (30-40)
        elif htftrend_bullish and rsi[i] > 30 and rsi[i] < 40 and rsi[i-1] <= 30:
            new_signal = SIZE_ENTRY
        # LONG ENTRY 3: 4h bullish + RSI neutral + price bounce from lower BB
        elif htftrend_bullish and rsi_neutral and price_near_lower and close[i] > close[i-1]:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + RSI overbought + price near upper BB + BB wide
        if htftrend_bearish and rsi_overbought and price_near_upper and bb_wide:
            new_signal = -SIZE_ENTRY
        # SHORT ENTRY 2: 4h bearish + RSI coming out of overbought (60-70)
        elif htftrend_bearish and rsi[i] < 70 and rsi[i] > 60 and rsi[i-1] >= 70:
            new_signal = -SIZE_ENTRY
        # SHORT ENTRY 3: 4h bearish + RSI neutral + price rejection from upper BB
        elif htftrend_bearish and rsi_neutral and price_near_upper and close[i] < close[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update stoploss (2*ATR trailing)
            current_stop = entry_price + (close[i] - entry_price) - 2.0 * atr[i]
            if close[i] > entry_price:
                current_stop = max(stoploss_price, close[i] - 2.0 * atr[i])
            
            if close[i] < stoploss_price:
                new_signal = SIZE_EXIT
            # Take profit at 2R
            elif not take_profit_price:
                risk = abs(entry_price - stoploss_price)
                if risk > 0 and (close[i] - entry_price) >= 2.0 * risk:
                    new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update stoploss (2*ATR trailing)
            if close[i] < entry_price:
                current_stop = min(stoploss_price, close[i] + 2.0 * atr[i])
            
            if close[i] > stoploss_price:
                new_signal = SIZE_EXIT
            # Take profit at 2R
            elif not take_profit_price:
                risk = abs(entry_price - stoploss_price)
                if risk > 0 and (entry_price - close[i]) >= 2.0 * risk:
                    new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            take_profit_price = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            take_profit_price = 0.0
        
        # Position closed
        if new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stoploss_price = 0.0
            take_profit_price = 0.0
        
        signals[i] = new_signal
    
    return signals