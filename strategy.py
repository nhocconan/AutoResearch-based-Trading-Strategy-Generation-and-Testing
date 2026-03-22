#!/usr/bin/env python3
"""
Experiment #008: 30m Donchian Breakout + 4h HMA Trend + Volume + Z-Score Mean Reversion
Hypothesis: 30m captures multi-day swings better than 15m (less noise) but trades more than 1h.
Donchian(20) breakouts with 4h HMA trend filter capture sustained moves. Volume confirmation
(taker_buy_ratio) filters false breakouts. Z-score(20) provides mean reversion entries when
ADX<20 (ranging market). 2.5*ATR stoploss for 30m bars. Conservative sizing (0.25-0.30).
CRITICAL: Multiple entry paths (4 long + 4 short) to ensure >=10 trades per symbol.
Looser filters than #002 to avoid 0-trade failure. Timeframe: 30m (REQUIRED), HTF: 4h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_hma_vol_zscore_atr_v1"
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
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / (std + 1e-10)
    return zscore

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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

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
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    zscore = calculate_zscore(close, 20)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    rsi = calculate_rsi(close, 14)
    
    # 30m HMA for additional trend confirmation
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - LOOSE filter to ensure trades
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_30m[i] > hma_30m[i-1] if i > 0 else False
        hma_falling = hma_30m[i] < hma_30m[i-1] if i > 0 else False
        
        # Donchian breakout signals (LOOSE - compare to 2 bars ago for confirmation)
        breakout_long = close[i] > donch_upper[i-2] if i > 1 else False
        breakout_short = close[i] < donch_lower[i-2] if i > 1 else False
        
        # Volume confirmation - LOOSE threshold
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        # ADX regime - LOOSE threshold
        trend_strong = adx[i] > 15
        trend_weak = adx[i] < 25
        
        # Z-score extremes - LOOSE threshold
        zscore_oversold = zscore[i] < -1.2
        zscore_overbought = zscore[i] > 1.2
        
        # RSI extremes for additional entries
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Fast HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        fast_rising = hma_30m_fast[i] > hma_30m_fast[i-1] if i > 0 else False
        fast_falling = hma_30m_fast[i] < hma_30m_fast[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (4 paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + 4h bullish + volume ok (primary trend entry)
        if breakout_long and htf_bullish and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + 30m HMA bullish + Fast HMA crossover up (trend continuation)
        elif htf_bullish and hma_30m_bullish and fast_above_slow and fast_rising:
            new_signal = SIZE_ENTRY
        
        # Path 3: Z-score oversold + 4h bullish (mean reversion in uptrend)
        elif zscore_oversold and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: RSI oversold + 4h bullish + ADX weak (dip buy in ranging uptrend)
        elif rsi_oversold and htf_bullish and trend_weak:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (4 paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + 4h bearish + volume ok (primary trend entry)
        if breakout_short and htf_bearish and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + 30m HMA bearish + Fast HMA crossover down (trend continuation)
        elif htf_bearish and hma_30m_bearish and fast_below_slow and fast_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Z-score overbought + 4h bearish (mean reversion in downtrend)
        elif zscore_overbought and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: RSI overbought + 4h bearish + ADX weak (rally sell in ranging downtrend)
        elif rsi_overbought and htf_bearish and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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