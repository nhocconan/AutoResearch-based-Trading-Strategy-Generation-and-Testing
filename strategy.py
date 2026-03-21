#!/usr/bin/env python3
"""
Experiment #372: 1d Donchian Breakout + Weekly HMA Trend + Choppiness Regime + ATR Stop
Hypothesis: Donchian Channel breakouts (20-period) are proven trend-following signals that work
well on daily timeframes for crypto. Weekly HMA provides higher-timeframe trend bias.
Choppiness Index filters out ranging markets where breakouts fail. RSI confirms momentum.
This combines Turtle Trading breakout logic with modern regime detection.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 20-50 trades total (daily = fewer but higher quality signals).
Key insight: Daily breakouts with weekly trend filter should capture major moves while avoiding
whipsaws in ranging markets. Loose RSI thresholds ensure minimum trade frequency.
Position sizing: 0.25 entry, 0.125 half (take profit), stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_chop_regime_rsi_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (avoid breakouts)
    CHOP < 38.2 = trending market (favor breakouts)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1])
            tr3 = np.abs(low[j] - close[j-1])
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period] = upper[period]
    lower[:period] = lower[period]
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    chop = calculate_choppiness(high, low, close, 14)
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Donchian middle line
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Choppiness regime filter
        is_trending = chop[i] < 55  # Loose filter to allow more trades
        is_ranging = chop[i] >= 55
        
        # Donchian breakout signals
        breakout_long = close[i] > donch_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donch_lower[i-1]  # Break below previous lower
        
        # Donchian position (already broken out)
        donch_bullish = close[i] > donch_mid[i]
        donch_bearish = close[i] < donch_mid[i]
        
        # RSI filter (LOOSE to ensure trade frequency on daily)
        rsi_ok_long = rsi[i] > 30  # Not deeply oversold
        rsi_ok_short = rsi[i] < 70  # Not deeply overbought
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 35 and rsi[i] < 80
        rsi_momentum_short = rsi[i] > 20 and rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Donchian breakout long + Weekly bullish + Trending + RSI ok
        if breakout_long and weekly_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Donchian bullish + Weekly bullish + RSI momentum (no breakout needed)
        elif donch_bullish and weekly_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian breakout long + RSI ok (weekly neutral ok in strong trend)
        elif breakout_long and is_trending and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: Donchian bullish alone (ensures minimum trade frequency)
        elif donch_bullish and rsi[i] > 35 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Quintenary: Breakout without regime filter (catches strong moves)
        elif breakout_long and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Donchian breakout short + Weekly bearish + Trending + RSI ok
        if breakout_short and weekly_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Donchian bearish + Weekly bearish + RSI momentum (no breakout needed)
        elif donch_bearish and weekly_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian breakout short + RSI ok (weekly neutral ok in strong trend)
        elif breakout_short and is_trending and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: Donchian bearish alone (ensures minimum trade frequency)
        elif donch_bearish and rsi[i] > 25 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quintenary: Breakout without regime filter (catches strong moves)
        elif breakout_short and rsi[i] < 60:
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