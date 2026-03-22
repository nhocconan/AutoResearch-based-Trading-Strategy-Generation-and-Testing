#!/usr/bin/env python3
"""
Experiment #015: 1h Vol Spike Mean Reversion + 4h Trend Filter + BB Bands + ATR Stop
Hypothesis: Volatility spikes (ATR(7)/ATR(30) > 1.8) indicate panic/exhaustion points.
Combined with Bollinger Band extremes and 4h trend filter, this captures mean reversion
in the direction of the higher timeframe trend. Asymmetric logic: only long when 4h bullish,
only short when 4h bearish. This avoids counter-trend trades that failed in 2022 crash.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25) controls DD.
2.5*ATR stoploss appropriate for 1h timeframe with vol spike entries.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volspike_bb_4h_hma_asymmetric_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma, upper, lower

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Volatility spike ratio
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    mask = atr_30 > 0
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    # Bollinger Bands
    bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    # RSI
    rsi = calculate_rsi(close, 14)
    
    # ADX for regime detection
    adx = calculate_adx(high, low, close, 14)
    
    # 1h HMA for additional trend confirmation
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - ASYMMETRIC LOGIC
        four_h_bullish = close[i] > hma_4h_aligned[i]
        four_h_bearish = close[i] < hma_4h_aligned[i]
        
        # Volatility spike detection
        vol_spike = atr_ratio[i] > 1.8
        
        # Bollinger Band positions
        price_near_lower = close[i] < bb_lower[i] * 1.005  # within 0.5% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.995  # within 0.5% of upper band
        price_below_bb = close[i] < bb_sma[i]
        price_above_bb = close[i] > bb_sma[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        hma_rising = hma_1h[i] > hma_1h[i-1] if i > 0 else False
        hma_falling = hma_1h[i] < hma_1h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # ADX regime
        trending = adx[i] > 20 if not np.isnan(adx[i]) else False
        ranging = adx[i] < 20 if not np.isnan(adx[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish - asymmetric) ===
        
        # Path 1: Vol spike + BB lower + 4h bullish + RSI oversold
        if four_h_bullish and vol_spike and price_near_lower and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + BB lower + RSI extreme oversold (stronger signal)
        elif four_h_bullish and price_near_lower and rsi_extreme_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + HMA rising + Fast HMA crossover up + RSI neutral
        elif four_h_bullish and hma_rising and fast_above_slow and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # Path 4: 4h bullish + Price below BB mid + RSI bouncing from oversold
        elif four_h_bullish and price_below_bb and rsi[i] < 45 and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h bullish + HMA bullish + RSI pullback to 45-55
        elif four_h_bullish and hma_1h_bullish and rsi[i] > 45 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (only when 4h bearish - asymmetric) ===
        
        # Path 1: Vol spike + BB upper + 4h bearish + RSI overbought
        if four_h_bearish and vol_spike and price_near_upper and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + BB upper + RSI extreme overbought (stronger signal)
        elif four_h_bearish and price_near_upper and rsi_extreme_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + HMA falling + Fast HMA crossover down + RSI neutral
        elif four_h_bearish and hma_falling and fast_below_slow and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 4h bearish + Price above BB mid + RSI dropping from overbought
        elif four_h_bearish and price_above_bb and rsi[i] > 55 and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h bearish + HMA bearish + RSI pullback to 45-55
        elif four_h_bearish and hma_1h_bearish and rsi[i] > 45 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr_14[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr_14[i]
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
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
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