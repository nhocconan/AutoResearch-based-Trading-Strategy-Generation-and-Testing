#!/usr/bin/env python3
"""
Experiment #430: 4h KAMA Adaptive Trend + Daily HMA + Choppiness Regime + RSI Momentum
Hypothesis: KAMA adapts to market efficiency (fast in trends, slow in ranges), combined with
Choppiness Index regime detection to switch between trend-following and mean-reversion modes.
Daily HMA provides long-term bias. Multiple entry paths ensure >=10 trades/symbol.
4h timeframe captures multi-day swings without excessive noise. CHOP filter avoids whipsaws.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 4h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_chop_regime_rsi_atr_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility != 0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest - lowest > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_line[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Choppiness regime
        is_trending = chop[i] < 38.2  # Trending market
        is_ranging = chop[i] > 61.8   # Ranging market
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # RSI momentum (RELAXED thresholds to ensure trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 65
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Trending regime + KAMA bullish + Daily bullish + RSI ok
        if is_trending and kama_bullish and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Trending regime + Above SMA50 + MACD bullish + RSI momentum
        elif is_trending and above_sma50 and macd_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: Ranging regime + KAMA bullish + RSI not overbought (mean reversion)
        elif is_ranging and kama_bullish and rsi[i] > 30 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Path 4: KAMA crossover (price crosses above KAMA) + Daily bullish
        elif kama_bullish and close[i-1] <= kama[i-1] and daily_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Simple trend - price > SMA50 + Daily bullish + RSI > 45
        elif above_sma50 and daily_bullish and rsi[i] > 45 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Path 6: MACD histogram turns positive + Daily bullish + RSI ok
        elif macd_bullish and macd_hist[i-1] <= 0 and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Trending regime + KAMA bearish + Daily bearish + RSI ok
        if is_trending and kama_bearish and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Trending regime + Below SMA50 + MACD bearish + RSI momentum
        elif is_trending and below_sma50 and macd_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Ranging regime + KAMA bearish + RSI not oversold (mean reversion)
        elif is_ranging and kama_bearish and rsi[i] > 30 and rsi[i] < 70:
            new_signal = -SIZE_ENTRY
        # Path 4: KAMA crossover (price crosses below KAMA) + Daily bearish
        elif kama_bearish and close[i-1] >= kama[i-1] and daily_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple trend - price < SMA50 + Daily bearish + RSI < 55
        elif below_sma50 and daily_bearish and rsi[i] < 55 and rsi[i] > 25:
            new_signal = -SIZE_ENTRY
        # Path 6: MACD histogram turns negative + Daily bearish + RSI ok
        elif macd_bearish and macd_hist[i-1] >= 0 and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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