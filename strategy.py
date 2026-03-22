#!/usr/bin/env python3
"""
Experiment #013: 15m MACD Momentum + 4h HMA Trend + RSI Filter + ATR Stop
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides robust trend filter.
MACD histogram crossovers give clear entry signals, RSI avoids overbought/oversold traps.
Multiple entry paths (MACD cross, MACD divergence, trend continuation) ensure >=10 trades.
Conservative sizing (0.25) with 2*ATR stoploss controls drawdown.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_macd_4h_hma_rsi_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m trend
        ema_21_bullish = close[i] > ema_21[i]
        ema_21_bearish = close[i] < ema_21[i]
        ema_50_bullish = close[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        ema_50_bearish = close[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # EMA crossover
        ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1] if i > 0 and not np.isnan(ema_50[i-1]) else False
        ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1] if i > 0 and not np.isnan(ema_50[i-1]) else False
        
        # MACD signals
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_long = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_short = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        macd_rising = macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_falling = macd_hist[i] < macd_hist[i-1] if i > 0 else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Bollinger Band position
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05 if bb_mid[i] > 0 else False
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bullish + MACD cross long + RSI bullish
        if hma_4h_bullish and macd_cross_long and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + EMA cross long + MACD bullish
        elif hma_4h_bullish and ema_cross_long and macd_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + MACD rising + RSI neutral + price > EMA21
        elif hma_4h_bullish and macd_rising and rsi_neutral and ema_21_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: 4h bullish + RSI oversold bounce + MACD not too bearish
        elif hma_4h_bullish and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            if macd_hist[i] > -0.001:  # MACD not deeply negative
                new_signal = SIZE_ENTRY
        
        # Path 5: EMA cross long + 4h not bearish + MACD improving
        elif ema_cross_long and not hma_4h_bearish and macd_hist[i] > macd_hist[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 6: Price near BB lower + 4h bullish + RSI oversold (mean reversion)
        elif price_near_lower and hma_4h_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bearish + MACD cross short + RSI bearish
        if hma_4h_bearish and macd_cross_short and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + EMA cross short + MACD bearish
        elif hma_4h_bearish and ema_cross_short and macd_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + MACD falling + RSI neutral + price < EMA21
        elif hma_4h_bearish and macd_falling and rsi_neutral and ema_21_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 4h bearish + RSI overbought drop + MACD not too bullish
        elif hma_4h_bearish and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            if macd_hist[i] < 0.001:  # MACD not deeply positive
                new_signal = -SIZE_ENTRY
        
        # Path 5: EMA cross short + 4h not bullish + MACD worsening
        elif ema_cross_short and not hma_4h_bullish and macd_hist[i] < macd_hist[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Price near BB upper + 4h bearish + RSI overbought (mean reversion)
        elif price_near_upper and hma_4h_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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