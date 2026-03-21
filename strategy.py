#!/usr/bin/env python3
"""
Experiment #407: 12h Donchian Breakout + Daily HMA Trend + ADX Filter + RSI Pullback + ATR Stop
Hypothesis: 12h timeframe captures medium-term trends better than 4h (less noise) and 1d (more signals).
Donchian breakout (20-period) identifies trend direction, Daily HMA provides long-term bias via mtf_data.
ADX(14) > 20 confirms trend strength (not too strict like ADX>40). RSI(14) pullback entries improve timing.
ATR(14) trailing stop at 2.0x for 12h timeframe. Position size 0.25 discrete with half-profit at 2R.
Key insight: 12h should generate 20-40 trades/year per symbol - enough for stats, few enough to minimize fees.
Multiple entry conditions ensure trade frequency across BTC/ETH/SOL. Daily HTF via mtf_data ensures no look-ahead.
Timeframe: 12h (REQUIRED for this experiment), HTF: 1d for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_hma_adx_rsi_pullback_atr_v1"
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
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    dx[:] = np.nan
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_raw[period*2:]
    
    return adx

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(close)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Donchian midline
    donch_mid = (donch_upper + donch_lower) / 2
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # ADX trend strength (not too strict - ADX>20 is enough for 12h)
        trend_strong = adx[i] > 20
        trend_very_strong = adx[i] > 25
        
        # Donchian breakout signals
        donch_bullish = close[i] > donch_mid[i]
        donch_bearish = close[i] < donch_mid[i]
        
        # Donchian upper/lower break (stronger signal)
        donch_break_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_break_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # RSI pullback conditions (loose to ensure trade frequency)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 70
        rsi_ok_short = rsi[i] > 30 and rsi[i] < 65
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # RSI pullback in uptrend
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 60
        rsi_pullback_short = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades on 12h) ===
        # Primary: Donchian bullish + Daily bullish + ADX strong + RSI ok
        if donch_bullish and daily_bullish and trend_strong and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Donchian breakout + Daily bullish + RSI momentum
        elif donch_break_long and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian bullish + ADX strong + RSI pullback (daily neutral ok)
        elif donch_bullish and trend_strong and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Daily bullish + Donchian bullish + RSI ok (ADX neutral)
        elif daily_bullish and donch_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Quintenary: Donchian breakout + RSI momentum (trend filter loose)
        elif donch_break_long and rsi_momentum_long and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades on 12h) ===
        # Primary: Donchian bearish + Daily bearish + ADX strong + RSI ok
        if donch_bearish and daily_bearish and trend_strong and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Donchian breakdown + Daily bearish + RSI momentum
        elif donch_break_short and daily_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian bearish + ADX strong + RSI pullback (daily neutral ok)
        elif donch_bearish and trend_strong and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Daily bearish + Donchian bearish + RSI ok (ADX neutral)
        elif daily_bearish and donch_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Quintenary: Donchian breakdown + RSI momentum (trend filter loose)
        elif donch_break_short and rsi_momentum_short and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest for 12h timeframe)
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
            
            # Calculate trailing stop (2.0*ATR from lowest for 12h timeframe)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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