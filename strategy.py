#!/usr/bin/env python3
"""
Experiment #264: 1d HMA Trend + RSI Pullback with Weekly Bias
Hypothesis: Daily timeframe needs simpler logic to generate sufficient trades. 
Previous multi-filter strategies failed with 0 trades. This uses:
- HMA(21) crossover for trend direction (faster response than EMA)
- RSI(14) pullback entries (40-60 range, not extremes - ensures trades)
- Weekly HMA(21) for macro bias (loose filter, not hard requirement)
- Volume confirmation via taker buy ratio
- ATR(14) trailing stop at 2.5*ATR
Key difference: Fewer filters, looser conditions to guarantee ≥10 trades.
Position sizing: 0.25 entry, 0.125 half at 2R profit. Target: Beat Sharpe=0.499.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_weekly_bias_volume_atr_v1"
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for crossover detection
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_48 = np.roll(hma_48, 1)
    prev_rsi = np.roll(rsi, 1)
    prev_hma_21[0] = hma_21[0]
    prev_hma_48[0] = hma_48[0]
    prev_rsi[0] = rsi[0]
    
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
        # Weekly trend bias (loose - just preference, not hard filter)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily HMA crossover signals (primary trend)
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        hma_cross_up = prev_hma_21[i] <= prev_hma_48[i] and hma_21[i] > hma_48[i]
        hma_cross_down = prev_hma_21[i] >= prev_hma_48[i] and hma_21[i] < hma_48[i]
        
        # Price relative to HMA
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # RSI signals (loose range to ensure trades)
        rsi_bullish = rsi[i] > 40
        rsi_bearish = rsi[i] < 60
        rsi_rising = rsi[i] > prev_rsi[i]
        rsi_falling = rsi[i] < prev_rsi[i]
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # HMA crossover with RSI confirmation
        if hma_cross_up:
            if rsi_bullish and (weekly_bullish or vol_bullish):
                new_signal = SIZE_ENTRY
        # Pullback to HMA in uptrend
        elif hma_bullish and price_below_hma:
            if prev_rsi[i] <= 45 and rsi[i] > 40 and rsi_rising:
                new_signal = SIZE_ENTRY
        # Price above HMA with momentum
        elif hma_bullish and price_above_hma:
            if rsi_bullish and rsi_rising and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # HMA crossover with RSI confirmation
        if hma_cross_down:
            if rsi_bearish and (weekly_bearish or vol_bearish):
                new_signal = -SIZE_ENTRY
        # Pullback to HMA in downtrend
        elif hma_bearish and price_above_hma:
            if prev_rsi[i] >= 55 and rsi[i] < 60 and rsi_falling:
                new_signal = -SIZE_ENTRY
        # Price below HMA with momentum
        elif hma_bearish and price_below_hma:
            if rsi_bearish and rsi_falling and vol_bearish:
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