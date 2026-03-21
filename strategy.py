#!/usr/bin/env python3
"""
Experiment #397: 15m Supertrend + 4h HMA Trend + 1h ADX Filter + RSI Pullback + ATR Stop
Hypothesis: 15m timeframe needs faster entries than daily strategies. Using 4h HMA for 
long-term trend bias (like the winning 12h strategy), 1h Supertrend for intermediate 
trend confirmation, and 15m RSI pullback for precise entry timing. ADX(14) > 20 filters 
out choppy periods. This should generate MORE trades than daily strategies while 
maintaining quality. ATR(14) stoploss at 2.0x for 15m timeframe (tighter than daily's 2.5x).
Position size 0.25 discrete. Target: Beat Sharpe=0.499 with higher trade frequency.
Timeframe: 15m (REQUIRED for this experiment), HTF: 1h and 4h via mtf_data helper.
Key insight: The winning strategy uses supertrend + HMA + RSI. Applying to 15m with 
proper MTF alignment should capture more opportunities while respecting trend direction.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_1h_adx_rsi_pullback_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        mid = (high[i] + low[i]) / 2
        upper = mid + multiplier * atr[i]
        lower = mid - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper
            direction[i] = 1
        else:
            # Update upper/lower based on previous direction
            if direction[i-1] == 1:
                upper = min(upper, supertrend[i-1])
                if close[i] > supertrend[i-1]:
                    supertrend[i] = upper
                    direction[i] = 1
                else:
                    supertrend[i] = lower
                    direction[i] = -1
            else:
                lower = max(lower, supertrend[i-1])
                if close[i] < supertrend[i-1]:
                    supertrend[i] = lower
                    direction[i] = -1
                else:
                    supertrend[i] = upper
                    direction[i] = 1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if i == period:
            plus_di[i] = 100 * np.sum(plus_dm[i-period+1:i+1]) / np.sum(tr[i-period+1:i+1]) if np.sum(tr[i-period+1:i+1]) > 0 else 0
            minus_di[i] = 100 * np.sum(minus_dm[i-period+1:i+1]) / np.sum(tr[i-period+1:i+1]) if np.sum(tr[i-period+1:i+1]) > 0 else 0
        else:
            plus_di[i] = (plus_di[i-1] * (period - 1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i-1] * (period - 1) + minus_dm[i]) / period
            
            tr_sum = np.sum(tr[i-period+1:i+1])
            if tr_sum > 0:
                plus_di[i] = 100 * plus_di[i] / tr_sum
                minus_di[i] = 100 * minus_di[i] / tr_sum
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for long-term trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h Supertrend for intermediate trend
    st_1h, st_dir_1h = calculate_supertrend(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 10, 3.0)
    st_1h_aligned = align_htf_to_ltf(prices, df_1h, st_1h)
    st_dir_1h_aligned = align_htf_to_ltf(prices, df_1h, st_dir_1h)
    
    # Calculate 1h ADX for trend strength
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Calculate 15m Supertrend for entry timing
    st_15m, st_dir_15m = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(st_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(st_1h_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend bias (long-term direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction (intermediate trend)
        st_1h_bullish = st_dir_1h_aligned[i] == 1
        st_1h_bearish = st_dir_1h_aligned[i] == -1
        
        # 15m Supertrend direction (short-term entry)
        st_15m_bullish = st_dir_15m[i] == 1
        st_15m_bearish = st_dir_15m[i] == -1
        
        # 1h ADX trend strength filter
        adx_strong = adx_1h_aligned[i] > 20  # Trending market
        adx_weak = adx_1h_aligned[i] <= 20   # Ranging market
        
        # RSI pullback conditions
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 60  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 40 and rsi[i] < 65  # Pullback in downtrend
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: All trends aligned + ADX strong + RSI pullback
        if hma_4h_bullish and st_1h_bullish and st_15m_bullish and adx_strong and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h + 1h bullish + 15m ST flip + RSI ok
        elif hma_4h_bullish and st_1h_bullish and st_15m_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + 15m ST bullish + RSI momentum (1h neutral)
        elif hma_4h_bullish and st_15m_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Quaternary: All ST aligned + RSI ok (4h neutral)
        elif st_1h_bullish and st_15m_bullish and rsi[i] > 45 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: 15m ST flip bullish + 4h bullish + RSI filter
        elif st_15m_bullish and st_dir_15m[i-1] == -1 and hma_4h_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: All trends aligned + ADX strong + RSI pullback
        if hma_4h_bearish and st_1h_bearish and st_15m_bearish and adx_strong and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h + 1h bearish + 15m ST flip + RSI ok
        elif hma_4h_bearish and st_1h_bearish and st_15m_bearish and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + 15m ST bearish + RSI momentum (1h neutral)
        elif hma_4h_bearish and st_15m_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: All ST aligned + RSI ok (4h neutral)
        elif st_1h_bearish and st_15m_bearish and rsi[i] > 30 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Quintenary: 15m ST flip bearish + 4h bearish + RSI filter
        elif st_15m_bearish and st_dir_15m[i-1] == 1 and hma_4h_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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