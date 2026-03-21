#!/usr/bin/env python3
"""
Experiment #367: 15m KAMA Adaptive Trend + 4h HMA Bias + Bollinger Regime + RSI Momentum
Hypothesis: 15m timeframe captures intraday swings better than 1h/4h. KAMA adapts to volatility
(less whipsaw in chop than EMA). 4h HMA provides trend bias via mtf_data helper. Bollinger Band
Width detects regime (squeeze=range, expansion=trend). RSI(7) confirms momentum without lag.
ATR(14) stoploss at 2.0x protects capital. Conservative sizing (0.25) controls drawdown.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 50-150 trades total across train+test.
Key insight: KAMA's Efficiency Ratio adapts to market conditions - faster in trends, slower in chop.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_hma_bollinger_regime_rsi_atr_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    volatility[0:period] = volatility[period]
    
    er = np.where(volatility > 0, change / volatility, 0.0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period-1] = close[period-1]
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma * 100  # Band Width as percentage
    return upper, lower, bw, sma

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    kama = calculate_kama(close, 10, 2, 30)
    bb_upper, bb_lower, bb_bw, bb_sma = calculate_bollinger(close, 20, 2.0)
    
    # KAMA fast line for crossover
    kama_fast = calculate_kama(close, 5, 2, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]) or np.isnan(bb_bw[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        trend_4h_bullish = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        
        # Bollinger regime detection
        # BW > 10 = trending, BW < 5 = squeeze/range
        regime_trending = bb_bw[i] > 8.0
        regime_range = bb_bw[i] < 5.0
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum (looser for 15m to ensure trades)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 70
        rsi_strong_long = rsi[i] > 40
        rsi_strong_short = rsi[i] < 60
        
        # Price relative to Bollinger bands
        price_near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross + 4h bullish + trending regime + RSI ok
        if kama_cross_long and trend_4h_bullish and regime_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA cross + 4h bullish + RSI strong (regime neutral)
        elif kama_cross_long and trend_4h_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Price above KAMA + 4h bullish + RSI ok (momentum continuation)
        elif price_above_kama and trend_4h_bullish and rsi[i] > 35 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA cross alone (ensures minimum trade frequency)
        elif kama_cross_long and rsi[i] > 25 and rsi[i] < 85:
            new_signal = SIZE_ENTRY
        # Range market mean reversion: price near lower band + RSI oversold
        elif regime_range and price_near_lower and rsi[i] < 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross + 4h bearish + trending regime + RSI ok
        if kama_cross_short and trend_4h_bearish and regime_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA cross + 4h bearish + RSI strong (regime neutral)
        elif kama_cross_short and trend_4h_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Price below KAMA + 4h bearish + RSI ok (momentum continuation)
        elif price_below_kama and trend_4h_bearish and rsi[i] > 25 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA cross alone (ensures minimum trade frequency)
        elif kama_cross_short and rsi[i] > 15 and rsi[i] < 75:
            new_signal = -SIZE_ENTRY
        # Range market mean reversion: price near upper band + RSI overbought
        elif regime_range and price_near_upper and rsi[i] > 65:
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
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
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
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals