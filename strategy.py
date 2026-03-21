#!/usr/bin/env python3
"""
Experiment #404: 30m Supertrend + 4h HMA Bias + RSI Pullback + ADX Filter + ATR Stop
Hypothesis: 30m timeframe needs stronger HTF filter to reduce noise. Using 4h HMA for 
trend bias (proven in best strategies), Supertrend(10,3) on 30m for clean entry signals,
RSI(14) pullback to 40-60 range for better entries, ADX(14)>20 to ensure trend strength.
Position size 0.25 (conservative for noisy 30m), ATR(14) stoploss at 2.5x.
Key insight from failures: 30m strategies failed due to too many conflicting filters.
This version loosens RSI range (35-65 instead of tight extremes) and allows entries
when ANY 2 of 3 conditions agree (HTF trend + Supertrend + RSI ok) to ensure trade frequency.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper for trend bias.
Target: Beat Sharpe=0.499 (current best), ensure ≥10 trades/symbol on train, ≥3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_4h_hma_rsi_adx_atr_v2"
timeframe = "30m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    hl2 = (high + low) / 2
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            # Update upper/lower bands
            if upper_band < supertrend[i-1] or close[i-1] > supertrend[i-1]:
                upper_band = supertrend[i-1]
            else:
                upper_band = hl2[i] + multiplier * atr[i]
            
            if lower_band > supertrend[i-1] or close[i-1] < supertrend[i-1]:
                lower_band = supertrend[i-1]
            else:
                lower_band = hl2[i] - multiplier * atr[i]
            
            # Determine direction
            if close[i] <= supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = 1
            else:
                supertrend[i] = lower_band
                direction[i] = -1
    
    return supertrend, direction

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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    avg_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = atr > 0
    plus_di[mask] = 100 * avg_plus_dm[mask] / atr[mask]
    minus_di[mask] = 100 * avg_minus_dm[mask] / atr[mask]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx[period*2:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period*2:]
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Supertrend direction on 30m
        st_bullish = st_direction[i] == -1  # -1 means price above supertrend
        st_bearish = st_direction[i] == 1   # 1 means price below supertrend
        
        # ADX trend strength filter (loose to ensure trades)
        adx_strong = adx[i] > 18  # Relaxed from 20 to get more trades
        
        # RSI pullback filter (wide range to ensure trades on 30m)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 70
        rsi_ok_short = rsi[i] > 30 and rsi[i] < 65
        
        # Count bullish/bearish signals (need 2 of 3 to agree)
        bullish_count = 0
        bearish_count = 0
        
        if hma_4h_bullish:
            bullish_count += 1
        if st_bullish:
            bullish_count += 1
        if rsi_ok_long:
            bullish_count += 1
        
        if hma_4h_bearish:
            bearish_count += 1
        if st_bearish:
            bearish_count += 1
        if rsi_ok_short:
            bearish_count += 1
        
        new_signal = 0.0
        
        # === LONG ENTRIES (need 2 of 3 conditions + ADX filter) ===
        # Primary: HTF bullish + Supertrend bullish + RSI ok
        if bullish_count >= 2 and adx_strong:
            new_signal = SIZE_ENTRY
        # Secondary: Strong Supertrend + HTF bullish (RSI optional)
        elif st_bullish and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend bullish + RSI strong + ADX ok
        elif st_bullish and rsi[i] > 40 and rsi[i] < 65 and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (need 2 of 3 conditions + ADX filter) ===
        # Primary: HTF bearish + Supertrend bearish + RSI ok
        if bearish_count >= 2 and adx_strong:
            new_signal = -SIZE_ENTRY
        # Secondary: Strong Supertrend + HTF bearish (RSI optional)
        elif st_bearish and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend bearish + RSI strong + ADX ok
        elif st_bearish and rsi[i] > 35 and rsi[i] < 60 and adx[i] > 15:
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