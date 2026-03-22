#!/usr/bin/env python3
"""
Experiment #001: 15m Multi-Timeframe Supertrend + 4h HMA Bias + RSI Pullback + ATR Stop
Hypothesis: 15m timeframe offers optimal balance between trade frequency and noise reduction.
4h HMA provides strong trend bias alignment, 1h Supertrend gives intermediate trend direction,
15m RSI pullbacks catch entries in direction of HTF trend. ADX filter ensures trend strength.
Multiple entry paths (5 long + 5 short) ensure >=10 trades per symbol. Conservative sizing (0.25)
controls drawdown. 2.0*ATR stoploss appropriate for 15m bars. Must beat Sharpe=0.499 baseline.
Timeframe: 15m (REQUIRED), HTF: 4h and 1h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_adx_atr_v1"
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
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    supertrend_1h, st_dir_1h = calculate_supertrend(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 10, 3.0)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    st_dir_1h_aligned = align_htf_to_ltf(prices, df_1h, st_dir_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_15m, st_dir_15m = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(st_dir_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(st_dir_15m[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction (intermediate HTF)
        st_1h_bullish = st_dir_1h_aligned[i] == 1
        st_1h_bearish = st_dir_1h_aligned[i] == -1
        
        # 15m Supertrend direction
        st_15m_bullish = st_dir_15m[i] == 1
        st_15m_bearish = st_dir_15m[i] == -1
        
        # Supertrend flip signals on 15m
        st_flip_long = st_dir_15m[i] == 1 and st_dir_15m[i-1] == -1 if i > 0 else False
        st_flip_short = st_dir_15m[i] == -1 and st_dir_15m[i-1] == 1 if i > 0 else False
        
        # EMA trend
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_very_strong = adx[i] > 25
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (5 paths for >=10 trades) ===
        
        # Path 1: All trends align + RSI pullback
        if hma_4h_bullish and st_1h_bullish and st_15m_bullish and trend_strong and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: 15m Supertrend flip long + HTF not bearish
        elif st_flip_long and not hma_4h_bearish and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h HMA bullish + EMA bullish + RSI oversold bounce
        elif hma_4h_bullish and ema_bullish and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1h Supertrend bullish + 15m Supertrend bullish + ADX building
        elif st_1h_bullish and st_15m_bullish and adx[i] > adx[i-1] if i > 0 else False and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h HMA bullish + 15m Supertrend bullish + RSI neutral
        elif hma_4h_bullish and st_15m_bullish and rsi_neutral and trend_strong:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (5 paths for >=10 trades) ===
        
        # Path 1: All trends align + RSI pullback
        if hma_4h_bearish and st_1h_bearish and st_15m_bearish and trend_strong and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 15m Supertrend flip short + HTF not bullish
        elif st_flip_short and not hma_4h_bullish and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h HMA bearish + EMA bearish + RSI overbought drop
        elif hma_4h_bearish and ema_bearish and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1h Supertrend bearish + 15m Supertrend bearish + ADX building
        elif st_1h_bearish and st_15m_bearish and adx[i] > adx[i-1] if i > 0 else False and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h HMA bearish + 15m Supertrend bearish + RSI neutral
        elif hma_4h_bearish and st_15m_bearish and rsi_neutral and trend_strong:
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