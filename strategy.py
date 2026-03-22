#!/usr/bin/env python3
"""
Experiment #007: 15m Fisher Transform + 4h HMA Trend + Volume Confirmation + ATR Stop
Hypothesis: 15m timeframe needs strong HTF filter to avoid noise. Fisher Transform (Ehlers)
excels at catching reversals in bear/range markets (2022 crash, 2025 bear). 4h HMA provides
trend bias without lag. Volume spike confirmation filters false breakouts. Conservative
sizing (0.25) with 2*ATR stoploss controls drawdown. Multiple entry paths ensure >=10 trades.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_volume_atr_v1"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for catching reversals in range/bear markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to -1 to +1 range
        value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * fisher[i - 1]
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        trigger[i] = fisher[i - 1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to recent average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    ema_4h_200 = calculate_ema(df_4h['close'].values, 200)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher(close, 9)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_ratio = calculate_volume_spike(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary trend filter
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i] and hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i] and hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # 4h major trend (200 EMA)
        htf_major_bull = close[i] > ema_4h_200_aligned[i]
        htf_major_bear = close[i] < ema_4h_200_aligned[i]
        
        # 15m EMA trend
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # Fisher Transform signals (reversal detection)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5 if i > 0 else False
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5 if i > 0 else False
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.5  # 50% above average
        
        # Bollinger Band position
        bb_lower_touch = close[i] <= bb_lower[i] * 1.002  # Near or below lower band
        bb_upper_touch = close[i] >= bb_upper[i] * 0.998  # Near or above upper band
        
        new_signal = 0.0
        
        # === LONG ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: 4h bullish + Fisher reversal long + RSI oversold
        if hma_4h_bullish and fisher_long and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + Fisher oversold + BB lower touch + volume
        elif hma_4h_bullish and fisher_oversold and bb_lower_touch and volume_confirmed:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h major bull + 15m EMA bullish + Fisher long + RSI neutral
        elif htf_major_bull and ema_bullish and fisher_long and rsi_neutral:
            new_signal = SIZE_ENTRY
        
        # Path 4: 4h bullish + RSI oversold + BB lower touch (mean reversion in uptrend)
        elif hma_4h_bullish and rsi_oversold and bb_lower_touch:
            new_signal = SIZE_ENTRY
        
        # Path 5: Fisher oversold + volume spike + RSI < 35 (strong reversal signal)
        elif fisher_oversold and volume_confirmed and rsi[i] < 35:
            new_signal = SIZE_ENTRY
        
        # Path 6: 4h not bearish + EMA bullish + Fisher > -1.0 (momentum continuation)
        elif not hma_4h_bearish and ema_bullish and fisher[i] > -1.0 and fisher[i] < 0.5:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: 4h bearish + Fisher reversal short + RSI overbought
        if hma_4h_bearish and fisher_short and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + Fisher overbought + BB upper touch + volume
        elif hma_4h_bearish and fisher_overbought and bb_upper_touch and volume_confirmed:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h major bear + 15m EMA bearish + Fisher short + RSI neutral
        elif htf_major_bear and ema_bearish and fisher_short and rsi_neutral:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 4h bearish + RSI overbought + BB upper touch (mean reversion in downtrend)
        elif hma_4h_bearish and rsi_overbought and bb_upper_touch:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Fisher overbought + volume spike + RSI > 65 (strong reversal signal)
        elif fisher_overbought and volume_confirmed and rsi[i] > 65:
            new_signal = -SIZE_ENTRY
        
        # Path 6: 4h not bullish + EMA bearish + Fisher < 1.0 (momentum continuation)
        elif not hma_4h_bullish and ema_bearish and fisher[i] < 1.0 and fisher[i] > -0.5:
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