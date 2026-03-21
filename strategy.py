#!/usr/bin/env python3
"""
Experiment #442: 4h KAMA + Fisher Transform + Daily HMA Bias + Volume Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than 
fixed-period MAs, reducing whipsaws in ranging markets. Fisher Transform catches reversals
with less lag than RSI. Daily HMA provides HTF trend bias. Volume confirmation filters
false breakouts. Multiple entry paths ensure >=10 trades per symbol. 2.5*ATR stoploss
for 4h timeframe balances protection with allowing normal volatility.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_daily_hma_volume_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er = change / volatility
        else:
            er = 0.0
        
        # Smoothing Constant
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i - period:i + 1] + low[i - period:i + 1]) / 2
        highest = np.max(hl2)
        lowest = np.min(hl2)
        
        if highest > lowest:
            x = (hl2[-1] - lowest) / (highest - lowest)
            x = np.clip(x, 0.001, 0.999)  # Avoid log(0)
            
            # Fisher Transform
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
            
            if i > period:
                fisher_signal[i] = fisher[i - 1]
        else:
            fisher[i] = fisher[i - 1] if i > period else 0.0
            fisher_signal[i] = fisher[i - 1] if i > period else 0.0
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i-1] if not np.isnan(kama[i-1]) else False
        kama_falling = kama[i] < kama[i-1] if not np.isnan(kama[i-1]) else False
        
        # Price relative to KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Fisher Transform signals (reversal detection)
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # RSI filter
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher cross up + Daily bullish + KAMA rising + Volume confirmed
        if fisher_cross_up and daily_bullish and kama_rising and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher oversold + Daily bullish + Above KAMA + ADX > 15
        elif fisher_oversold and daily_bullish and above_kama and adx[i] > 15:
            new_signal = SIZE_ENTRY
        # Path 3: Price above KAMA + KAMA rising + Daily bullish + RSI 40-65
        elif above_kama and kama_rising and daily_bullish and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 4: DI bullish + Daily bullish + Volume confirmed + ADX rising
        elif di_bullish and daily_bullish and volume_confirmed and adx[i] > adx[i-1] and adx[i] > 18:
            new_signal = SIZE_ENTRY
        # Path 5: Fisher cross up + Above KAMA + Volume confirmed + RSI > 35
        elif fisher_cross_up and above_kama and volume_confirmed and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Path 6: KAMA rising + Daily bullish + ADX > 20 + RSI 45-65
        elif kama_rising and daily_bullish and adx[i] > 20 and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher cross down + Daily bearish + KAMA falling + Volume confirmed
        if fisher_cross_down and daily_bearish and kama_falling and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher overbought + Daily bearish + Below KAMA + ADX > 15
        elif fisher_overbought and daily_bearish and below_kama and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        # Path 3: Price below KAMA + KAMA falling + Daily bearish + RSI 35-60
        elif below_kama and kama_falling and daily_bearish and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 4: DI bearish + Daily bearish + Volume confirmed + ADX rising
        elif di_bearish and daily_bearish and volume_confirmed and adx[i] > adx[i-1] and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        # Path 5: Fisher cross down + Below KAMA + Volume confirmed + RSI < 65
        elif fisher_cross_down and below_kama and volume_confirmed and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 6: KAMA falling + Daily bearish + ADX > 20 + RSI 35-55
        elif kama_falling and daily_bearish and adx[i] > 20 and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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