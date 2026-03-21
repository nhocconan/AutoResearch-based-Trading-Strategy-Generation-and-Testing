#!/usr/bin/env python3
"""
Experiment #265: 15m RSI Mean Reversion with 1h/4h HMA Trend Filter
Hypothesis: 15m timeframe is too noisy for trend-following (see exp#259, #260 failures).
Mean reversion works better on lower timeframes. Using RSI(7) for faster response on 15m,
with 1h HMA for immediate trend filter and 4h HMA for macro bias. Simple entry conditions:
RSI<20 long when both HTF bullish, RSI>80 short when both HTF bearish. This avoids the
complex multi-filter approaches that resulted in 0 trades (exp#254, #257). Position sizing:
0.25 entry, stoploss at 2.0*ATR. Target: Generate 50+ trades with Sharpe>0.5.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_mr_1h_4h_hma_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI indicator with shorter period for 15m."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion confirmation."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    zscore = calculate_zscore(close, 20)
    
    # Track previous RSI for momentum confirmation
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (both must agree for strong signal)
        hma_1h_bullish = close[i] > hma_1h_aligned[i]
        hma_1h_bearish = close[i] < hma_1h_aligned[i]
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Strong trend when both HTF agree
        strong_bullish = hma_1h_bullish and hma_4h_bullish
        strong_bearish = hma_1h_bearish and hma_4h_bearish
        
        # RSI mean reversion signals (looser thresholds for more trades)
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        rsi_rising = rsi[i] > prev_rsi[i]
        rsi_falling = rsi[i] < prev_rsi[i]
        
        # Z-score confirmation (extreme mean reversion)
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # RSI oversold in strong uptrend (mean reversion pullback)
        if rsi_oversold and strong_bullish:
            if rsi_rising or zscore_extreme_low:
                new_signal = SIZE_ENTRY
        
        # RSI very oversold even without strong trend (deep pullback)
        elif rsi[i] < 15 and hma_1h_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # RSI overbought in strong downtrend (mean reversion rally)
        if rsi_overbought and strong_bearish:
            if rsi_falling or zscore_extreme_high:
                new_signal = -SIZE_ENTRY
        
        # RSI very overbought even without strong trend (sharp rally)
        elif rsi[i] > 85 and hma_1h_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = SIZE_EXIT
            
            # Take profit at RSI>60 (mean reversion complete)
            elif rsi[i] > 60 and not new_signal:
                new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = SIZE_EXIT
            
            # Take profit at RSI<40 (mean reversion complete)
            elif rsi[i] < 40 and not new_signal:
                new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.0 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.0 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                trailing_stop = close[i] - 2.0 * atr[i]
                highest_close = close[i]
                lowest_close = 0.0
            else:
                trailing_stop = close[i] + 2.0 * atr[i]
                lowest_close = close[i]
                highest_close = 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals