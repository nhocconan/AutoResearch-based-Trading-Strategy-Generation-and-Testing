#!/usr/bin/env python3
"""
Experiment #360: 1d KAMA Trend + Weekly HMA Bias + RSI Pullback + Choppiness Regime + ATR Stop
Hypothesis: Daily timeframe captures major trend shifts with less noise than 12h/4h.
KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA for trend direction.
Weekly HMA provides macro trend confirmation via mtf_data helper.
Choppiness Index filters range vs trend regimes to adapt entry logic.
RSI pullback entries (not breakouts) work better on 1d for risk/reward.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades total across train+test.
Key insight: 1d needs looser entry filters than lower TFs to ensure trade frequency.
Multiple OR conditions ensure minimum trades while maintaining quality.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_rsi_chop_regime_atr_v1"
timeframe = "1d"
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
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Efficiency Ratio
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility == 0:
            er = 0
        else:
            er = change / volatility
        
        # Smoothing constant
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        range_hl = highest_high - lowest_low
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Daily trend via KAMA
        daily_bullish = close[i] > kama[i]
        daily_bearish = close[i] < kama[i]
        
        # KAMA slope (trend strength)
        kama_slope_up = kama[i] > kama[i - 5] if i >= 5 else False
        kama_slope_down = kama[i] < kama[i - 5] if i >= 5 else False
        
        # Choppiness regime
        is_trending = chop[i] < 45.0  # Lower threshold for more trend signals
        is_ranging = chop[i] > 55.0  # Lower threshold for more range signals
        
        # RSI levels (LOOSE for 1d to ensure trades)
        rsi_ok_long = rsi[i] < 55  # Not overbought
        rsi_ok_short = rsi[i] > 45  # Not oversold
        rsi_pullback_long = rsi[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55  # Pullback in downtrend
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Trending + Weekly bullish + Daily bullish + RSI pullback
        if is_trending and weekly_bullish and daily_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + KAMA slope up + RSI ok
        elif weekly_bullish and kama_slope_up and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Ranging + Weekly bullish + RSI low (mean reversion)
        elif is_ranging and weekly_bullish and rsi[i] < 40:
            new_signal = SIZE_ENTRY
        # Quaternary: Ensure trade frequency - Weekly bullish + KAMA bullish
        elif weekly_bullish and daily_bullish and rsi[i] > 30 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: KAMA cross up (momentum)
        elif close[i] > kama[i] and close[i - 1] <= kama[i - 1] and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Trending + Weekly bearish + Daily bearish + RSI pullback
        if is_trending and weekly_bearish and daily_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + KAMA slope down + RSI ok
        elif weekly_bearish and kama_slope_down and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Ranging + Weekly bearish + RSI high (mean reversion)
        elif is_ranging and weekly_bearish and rsi[i] > 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: Ensure trade frequency - Weekly bearish + KAMA bearish
        elif weekly_bearish and daily_bearish and rsi[i] > 30 and rsi[i] < 70:
            new_signal = -SIZE_ENTRY
        # Quintenary: KAMA cross down (momentum)
        elif close[i] < kama[i] and close[i - 1] >= kama[i - 1] and rsi[i] < 65:
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