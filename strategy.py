#!/usr/bin/env python3
"""
Experiment #166: 4h Fisher Transform Reversal with Daily/Weekly HMA Trend Filter
Hypothesis: 4h timeframe captures multi-day swings. Ehlers Fisher Transform excels
at identifying turning points in bear/range markets (2022 crash, 2025 consolidation).
Daily HMA provides major trend bias, Weekly HMA confirms macro direction.
KAMA adapts to market noise (fast in trends, slow in ranges). Entry on Fisher
extreme reversals (-1.5/+1.5 thresholds) with HTF trend confirmation.
This targets reversals during 2022 crash and 2025 bear market rallies.
Position sizing: 0.25 entry, stoploss at 2.5*ATR. Discrete levels minimize fees.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_daily_weekly_hma_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise: fast in trends, slow in ranges.
    Reference: Perry Kaufman, "Trading Systems and Methods"
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[:er_period] = np.nan
    
    er = np.where(volatility > 0, change / volatility, 0.0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Identifies turning points by normalizing price to Gaussian distribution.
    Reference: John Ehlers, "Rocket Science for Traders"
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_hl = hh - ll
        if range_hl < 1e-10:
            range_hl = 1e-10
        
        # Normalize price to 0-1 range
        value = (2.0 * (close[i] - ll) / range_hl) - 1.0
        value = np.clip(value, -0.999, 0.999)
        
        # Apply Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value))
        
        # Smooth with previous value
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=20)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
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
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # 4h trend via KAMA
        trend_bullish = kama_20[i] > kama_50[i]
        trend_bearish = kama_20[i] < kama_50[i]
        
        # Fisher Transform signals
        fisher_long = fisher[i] < -1.5 and fisher_signal[i] >= -1.5
        fisher_short = fisher[i] > 1.5 and fisher_signal[i] <= 1.5
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else False
        
        # MACD confirmation
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        new_signal = 0.0
        
        # === FISHER REVERSAL LONG ===
        # Entry when Fisher at extreme low and crossing up
        if fisher_long or fisher_cross_up:
            # Require either HTF bullish or 4h trend bullish
            if daily_bullish or weekly_bullish or trend_bullish:
                # Additional confirmation: RSI oversold or MACD positive
                if rsi_oversold or macd_positive:
                    new_signal = SIZE_ENTRY
        
        # === FISHER REVERSAL SHORT ===
        # Entry when Fisher at extreme high and crossing down
        elif fisher_short or fisher_cross_down:
            # Require either HTF bearish or 4h trend bearish
            if daily_bearish or weekly_bearish or trend_bearish:
                # Additional confirmation: RSI overbought or MACD negative
                if rsi_overbought or macd_negative:
                    new_signal = -SIZE_ENTRY
        
        # === KAMA TREND FOLLOWING ===
        # KAMA crossover with HTF confirmation
        if new_signal == 0.0:
            kama_cross_up = kama_20[i] > kama_50[i] and kama_20[i-1] <= kama_50[i-1]
            kama_cross_down = kama_20[i] < kama_50[i] and kama_20[i-1] >= kama_50[i-1]
            
            if kama_cross_up and (daily_bullish or weekly_bullish):
                if macd_bullish or rsi_rising:
                    new_signal = SIZE_ENTRY
            
            elif kama_cross_down and (daily_bearish or weekly_bearish):
                if macd_bearish or rsi_falling:
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