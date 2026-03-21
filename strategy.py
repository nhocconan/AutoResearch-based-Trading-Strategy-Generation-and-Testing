#!/usr/bin/env python3
"""
Experiment #257: 12h KAMA Trend + ROC Momentum with Daily HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than 
fixed EMAs, reducing whipsaw in choppy conditions. ROC(10) momentum confirms trend strength.
Daily HMA provides primary trend bias (proven in current best strategy). Simpler entry logic
than MACD/Donchian combo to ensure sufficient trades. Volume ratio as secondary confirmation.
Position sizing: 0.28 entry, 0.14 half at 2R profit. Stoploss: 2.5*ATR trailing stop.
Key difference from current best: KAMA crossover instead of Supertrend, ROC instead of RSI.
Target: Beat Sharpe=0.499 with fewer whipsaws in range markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_roc_daily_hma_volume_atr_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0:period] = np.abs(close[0:period] - close[0])
    
    er = np.where(volatility > 0, change / volatility, 0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_roc(close, period=10):
    """Calculate Rate of Change momentum indicator."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    roc = roc.fillna(0).values
    return roc

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    kama_fast = calculate_kama(close, period=5, fast=2, slow=20)
    roc = calculate_roc(close, 10)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for crossover detection
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_kama_fast = np.roll(kama_fast, 1)
    prev_kama_fast[0] = kama_fast[0]
    prev_roc = np.roll(roc, 1)
    prev_roc[0] = roc[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):
        # HTF trend filter (Daily HMA - proven in current best)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA crossover signals (adaptive trend)
        kama_cross_up = prev_kama[i] <= close[i] and kama[i] > close[i]
        kama_cross_down = prev_kama[i] >= close[i] and kama[i] < close[i]
        
        # KAMA fast/slow crossover
        kama_fs_cross_up = prev_kama_fast[i] <= prev_kama[i] and kama_fast[i] > kama[i]
        kama_fs_cross_down = prev_kama_fast[i] >= prev_kama[i] and kama_fast[i] < kama[i]
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # ROC momentum
        roc_positive = roc[i] > 0
        roc_negative = roc[i] < 0
        roc_strong_long = roc[i] > 2.0
        roc_strong_short = roc[i] < -2.0
        roc_improving = roc[i] > prev_roc[i]
        roc_worsening = roc[i] < prev_roc[i]
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA crossover with trend and momentum
        if kama_cross_up or kama_fs_cross_up:
            if daily_bullish and (roc_positive or vol_bullish):
                new_signal = SIZE_ENTRY
            elif price_above_kama and roc_improving:
                new_signal = SIZE_ENTRY
        
        # Price above KAMA with momentum in uptrend
        elif price_above_kama and daily_bullish:
            if roc_strong_long or (roc_positive and vol_bullish):
                new_signal = SIZE_ENTRY
        
        # Pullback to KAMA in uptrend
        elif daily_bullish and price_above_kama:
            if prev_kama[i] > close[i] and kama[i] <= close[i]:
                if roc_improving or vol_bullish:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA crossover with trend and momentum
        if kama_cross_down or kama_fs_cross_down:
            if daily_bearish and (roc_negative or vol_bearish):
                new_signal = -SIZE_ENTRY
            elif price_below_kama and roc_worsening:
                new_signal = -SIZE_ENTRY
        
        # Price below KAMA with momentum in downtrend
        elif price_below_kama and daily_bearish:
            if roc_strong_short or (roc_negative and vol_bearish):
                new_signal = -SIZE_ENTRY
        
        # Pullback to KAMA in downtrend
        elif daily_bearish and price_below_kama:
            if prev_kama[i] < close[i] and kama[i] >= close[i]:
                if roc_worsening or vol_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
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
            
            # Calculate trailing stop (2.5*ATR from lowest)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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