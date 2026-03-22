#!/usr/bin/env python3
"""
Experiment #492: 1d Supertrend + Weekly HMA Bias + RSI Pullback + ADX Filter + ATR Stop
Hypothesis: Daily timeframe provides cleaner signals with less noise than lower TFs.
Weekly HMA gives strong HTF trend bias alignment. Supertrend provides clear trend direction.
RSI pullback entries catch dips in uptrends (and rallies in downtrends) for better entry timing.
ADX filter ensures we only trade when trend has meaningful strength (>18 for daily).
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.28) controls DD.
3.5*ATR stoploss appropriate for daily bars (wider than 12h due to larger daily volatility).
Timeframe: 1d (REQUIRED for this experiment), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_weekly_hma_rsi_adx_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR bands.
    Returns: supertrend values, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
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
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Daily HMA for additional trend confirmation
    hma_1d = calculate_hma(close, 21)
    hma_1d_fast = calculate_hma(close, 10)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1 if i > 0 else False
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1 if i > 0 else False
        
        # Daily HMA trend
        hma_1d_bullish = close[i] > hma_1d[i]
        hma_1d_bearish = close[i] < hma_1d[i]
        hma_rising = hma_1d[i] > hma_1d[i-1] if i > 0 else False
        hma_falling = hma_1d[i] < hma_1d[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_1d_fast[i] > hma_1d[i]
        fast_below_slow = hma_1d_fast[i] < hma_1d[i]
        
        # ADX trend strength (lower threshold for daily to get more trades)
        trend_strong = adx[i] > 18
        trend_very_strong = adx[i] > 25
        
        # RSI zones (wider for daily timeframe)
        rsi_oversold = rsi[i] < 32
        rsi_overbought = rsi[i] > 68
        rsi_pullback_long = rsi[i] > 38 and rsi[i] < 52
        rsi_pullback_short = rsi[i] > 48 and rsi[i] < 62
        rsi_neutral = rsi[i] > 38 and rsi[i] < 62
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Supertrend bullish + Weekly bullish + ADX strong + RSI pullback
        if st_bullish and weekly_bullish and trend_strong and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: Supertrend flip long + Weekly not bearish + ADX ok
        elif st_flip_long and not weekly_bearish and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # Path 3: Weekly bullish + Daily HMA bullish + Fast HMA crossover up
        elif weekly_bullish and hma_1d_bullish and fast_above_slow and hma_1d_fast[i] > hma_1d_fast[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 4: Supertrend bullish + RSI oversold bounce (mean reversion in trend)
        elif st_bullish and rsi_oversold and i > 0 and rsi[i] > rsi[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 5: Weekly bullish + HMA rising + RSI neutral + ADX building
        elif weekly_bullish and hma_rising and rsi_neutral and i > 0 and adx[i] > adx[i-1]:
            new_signal = SIZE_ENTRY
        
        # Path 6: Supertrend bullish + Daily HMA bullish + ADX strong (simple trend follow)
        elif st_bullish and hma_1d_bullish and trend_strong:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Supertrend bearish + Weekly bearish + ADX strong + RSI pullback
        if st_bearish and weekly_bearish and trend_strong and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Supertrend flip short + Weekly not bullish + ADX ok
        elif st_flip_short and not weekly_bullish and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Weekly bearish + Daily HMA bearish + Fast HMA crossover down
        elif weekly_bearish and hma_1d_bearish and fast_below_slow and hma_1d_fast[i] < hma_1d_fast[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Supertrend bearish + RSI overbought drop (mean reversion in trend)
        elif st_bearish and rsi_overbought and i > 0 and rsi[i] < rsi[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Weekly bearish + HMA falling + RSI neutral + ADX building
        elif weekly_bearish and hma_falling and rsi_neutral and i > 0 and adx[i] > adx[i-1]:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Supertrend bearish + Daily HMA bearish + ADX strong (simple trend follow)
        elif st_bearish and hma_1d_bearish and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3.5*ATR for daily timeframe)
            current_stop = highest_close - 3.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3.5*ATR for daily timeframe)
            current_stop = lowest_close + 3.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.5 * atr[i]
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
            trailing_stop = close[i] - 3.5 * atr[i] if position_side > 0 else close[i] + 3.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.5 * atr[i] if position_side > 0 else close[i] + 3.5 * atr[i]
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