#!/usr/bin/env python3
"""
Experiment #471: 1h HMA Trend + 4h Bias + RSI(7) Pullback + Volume Confirm + ATR Stop
Hypothesis: 1h primary with 4h HTF provides optimal balance - 4h trend is responsive enough
for 1h entries while filtering noise. RSI(7) faster than RSI(14) catches pullbacks earlier.
Volume confirmation (taker_buy_ratio) filters false breakouts. 2.0*ATR stop tighter than 2.5*ATR.
Multiple entry paths ensure >=10 trades requirement while maintaining quality.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_4h_bias_rsi7_volume_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI indicator with shorter period for faster signals."""
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
    """Calculate taker buy volume ratio (0-1, >0.5 = buying pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 1h timeframe
    hma_slope = calculate_slope(hma_1h, lookback=5)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_slope[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # Volume confirmation (>0.55 = strong buying, <0.45 = strong selling)
        vol_buying = vol_ratio[i] > 0.55
        vol_selling = vol_ratio[i] < 0.45
        
        # RSI pullback zones (RSI(7) - faster than RSI(14))
        rsi_pullback_long = rsi[i] > 30 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 70
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 1h bullish + RSI pullback + HMA rising
        if htf_bullish and hma_1h_bullish and rsi_pullback_long and hma_rising:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + Fast HMA above slow + Volume buying + RSI > 35
        elif htf_bullish and fast_above_slow and vol_buying and rsi[i] > 35 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 3: 1h bullish + HMA rising + RSI oversold (deep pullback)
        elif hma_1h_bullish and hma_rising and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + 1h bullish + Fast HMA crossover up + Volume
        elif htf_bullish and hma_1h_bullish and fast_above_slow and vol_buying and hma_1h_fast[i] > hma_1h_fast[i-1]:
            new_signal = SIZE_ENTRY
        # Path 5: Price above both HMA + RSI 40-50 (consolidation breakout)
        elif close[i] > hma_1h[i] and close[i] > hma_4h_aligned[i] and rsi[i] > 40 and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        # Path 6: 4h bullish + RSI crossing up from oversold
        elif htf_bullish and rsi[i] > 35 and rsi[i-1] <= 35:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 1h bearish + RSI pullback + HMA falling
        if htf_bearish and hma_1h_bearish and rsi_pullback_short and hma_falling:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + Fast HMA below slow + Volume selling + RSI < 65
        elif htf_bearish and fast_below_slow and vol_selling and rsi[i] > 45 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 3: 1h bearish + HMA falling + RSI overbought (rally short)
        elif hma_1h_bearish and hma_falling and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + 1h bearish + Fast HMA crossover down + Volume
        elif htf_bearish and hma_1h_bearish and fast_below_slow and vol_selling and hma_1h_fast[i] < hma_1h_fast[i-1]:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below both HMA + RSI 50-60 (consolidation breakdown)
        elif close[i] < hma_1h[i] and close[i] < hma_4h_aligned[i] and rsi[i] > 50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 6: 4h bearish + RSI crossing down from overbought
        elif htf_bearish and rsi[i] < 65 and rsi[i-1] >= 65:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe - tighter than 2.5*ATR)
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
            
            # Calculate trailing stop (2.0*ATR for 1h timeframe)
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