#!/usr/bin/env python3
"""
Experiment #165: 1h HMA Trend + 4h HTF Filter + RSI Pullback
Hypothesis: 1h timeframe captures intraday swings while 4h HMA provides 
stable trend bias. RSI pullback entries (35/65 thresholds) ensure sufficient
trades while avoiding noise. ATR stoploss at 2.0*ATR controls drawdown.
Simplified from failed complex regime strategies - fewer filters = more trades.
Position sizing: 0.30 entry, discrete levels to minimize fee churn.
Key insight from failures: CRSI/complex regimes fail on short TFs, simple trend+pullback works.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_4h_trend_rsi_pullback_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (4h and 12h)
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        hma_12h_bullish = hma_12h_aligned[i] > 0 and close[i] > hma_12h_aligned[i]
        hma_12h_bearish = hma_12h_aligned[i] > 0 and close[i] < hma_12h_aligned[i]
        
        # 1h trend
        trend_bullish = hma_20[i] > hma_50[i] and hma_20[i] > sma_200[i]
        trend_bearish = hma_20[i] < hma_50[i] and hma_20[i] < sma_200[i]
        
        # RSI signals (loosened thresholds for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 35 <= rsi[i] <= 65
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] <= 20
        
        new_signal = 0.0
        
        # === TREND FOLLOWING LONG ===
        # Entry: 4h bullish + 1h trend + RSI pullback
        if hma_4h_bullish and trend_bullish:
            if rsi_oversold and rsi_rising:
                new_signal = SIZE_ENTRY
            elif rsi[i] < 50 and rsi_rising and trend_strong:
                new_signal = SIZE_ENTRY
        
        # === TREND FOLLOWING SHORT ===
        # Entry: 4h bearish + 1h trend + RSI pullback
        elif hma_4h_bearish and trend_bearish:
            if rsi_overbought and rsi_falling:
                new_signal = -SIZE_ENTRY
            elif rsi[i] > 50 and rsi_falling and trend_strong:
                new_signal = -SIZE_ENTRY
        
        # === MEAN REVERSION (when HTF and 1h disagree = range) ===
        if new_signal == 0.0 and trend_weak:
            # Long: RSI very oversold + price below 4h HMA
            if rsi[i] < 30 and close[i] < hma_4h_aligned[i] * 0.98:
                new_signal = SIZE_ENTRY * 0.5
            # Short: RSI very overbought + price above 4h HMA
            elif rsi[i] > 70 and close[i] > hma_4h_aligned[i] * 1.02:
                new_signal = -SIZE_ENTRY * 0.5
        
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
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals