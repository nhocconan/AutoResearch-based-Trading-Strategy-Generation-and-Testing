#!/usr/bin/env python3
"""
Experiment #480: 1d Donchian Breakout + Weekly HMA Bias + ADX-RSI Filter + ATR Stop
Hypothesis: Daily Donchian channels (20-period) capture sustained trends while weekly 
HMA provides higher-timeframe bias. ADX>25 filters weak trends, RSI 40-60 zone avoids 
extreme entries. Multiple entry paths (breakout + pullback) ensure >=10 trades. 
Daily timeframe reduces noise and fee impact vs lower TFs. 3*ATR stop appropriate 
for daily volatility. This should work in both 2021-2022 bull and 2025 bear markets.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_adx_rsi_atr_v1"
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
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    
    return upper, lower

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
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Also calculate shorter Donchian for pullback entries
    donchian_upper_10, donchian_lower_10 = calculate_donchian(high, low, 10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # ADX trend strength (lower threshold for daily TF)
        trend_strong = adx[i] > 25
        trend_moderate = adx[i] > 20
        
        # RSI zones (avoid extremes for trend following)
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 60
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Price position in channel
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 0 and not np.isnan(donchian_upper[i]):
            channel_pct = (close[i] - donchian_lower[i]) / channel_range
        else:
            channel_pct = 0.5
        
        in_upper_half = channel_pct > 0.5
        in_lower_half = channel_pct < 0.5
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bullish + Donchian breakout + ADX strong + RSI ok
        if weekly_bullish and breakout_long and trend_strong and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 2: Weekly bullish + Price in upper half + DI bullish + ADX moderate
        elif weekly_bullish and in_upper_half and di_bullish and trend_moderate:
            new_signal = SIZE_ENTRY
        # Path 3: Donchian breakout + Weekly bullish + RSI pullback zone
        elif breakout_long and weekly_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Path 4: Price > weekly HMA + ADX > 20 + DI bullish + RSI 45-65
        elif weekly_bullish and adx[i] > 20 and di_bullish and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Path 5: Breakout on 10-day Donchian + Weekly bullish + ADX > 18
        elif (close[i] > donchian_upper_10[i-1] if not np.isnan(donchian_upper_10[i-1]) else False) and weekly_bullish and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bearish + Donchian breakout + ADX strong + RSI ok
        if weekly_bearish and breakout_short and trend_strong and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Weekly bearish + Price in lower half + DI bearish + ADX moderate
        elif weekly_bearish and in_lower_half and di_bearish and trend_moderate:
            new_signal = -SIZE_ENTRY
        # Path 3: Donchian breakdown + Weekly bearish + RSI pullback zone
        elif breakout_short and weekly_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Path 4: Price < weekly HMA + ADX > 20 + DI bearish + RSI 35-55
        elif weekly_bearish and adx[i] > 20 and di_bearish and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: Breakdown on 10-day Donchian + Weekly bearish + ADX > 18
        elif (close[i] < donchian_lower_10[i-1] if not np.isnan(donchian_lower_10[i-1]) else False) and weekly_bearish and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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