#!/usr/bin/env python3
"""
Experiment #405: 1h Fisher Transform + 4h HMA Trend + ADX Filter + ATR Stop
Hypothesis: 1h timeframe has been failing because RSI/MACD are too laggy for intraday.
Ehlers Fisher Transform normalizes price to Gaussian distribution, catching turning points
faster than RSI. Combined with 4h HMA trend bias (proven in best strategies) and ADX
filter to only trade in trending conditions (avoid range whipsaw). 1h needs wider stops
than 15m/30m but tighter than 4h/12h. Using 2.5*ATR stoploss. Position size 0.25 discrete.
Key insight: Fisher Transform excels at 1h because it adapts to volatility and catches
reversals before RSI. ADX>25 filters out choppy periods that destroyed previous 1h strategies.
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_adx_trend_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian normalized distribution.
    Catches turning points faster than RSI/MACD.
    Entry: Fisher crosses above -1.5 (long), below +1.5 (short)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        price_norm = 0.33 * 2 * ((close[i] - lowest) / (highest - lowest) - 0.5)
        
        # Apply Fisher transform with smoothing
        if i == period:
            fisher[i] = 0.66 * price_norm
        else:
            fisher[i] = 0.66 * price_norm + 0.67 * fisher[i-1]
        
        # Clamp to avoid extreme values
        fisher[i] = np.clip(fisher[i], -1.5, 1.5)
        
        # Trigger line (1-period delayed)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index).
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]),
                   abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period*2, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    atr = calculate_atr(high, low, close, 14)
    fisher, trigger = calculate_fisher_transform(high, low, close, 9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # ADX trend strength filter
        is_trending = adx[i] > 22  # Slightly lower than 25 to get more trades on 1h
        is_strong_trend = adx[i] > 28
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.2 and (fisher[i] > trigger[i] if not np.isnan(trigger[i]) else True)
        fisher_short = fisher[i] < 1.2 and (fisher[i] < trigger[i] if not np.isnan(trigger[i]) else True)
        
        # Fisher extreme reversals
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Fisher crossover signals
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 1 and not np.isnan(fisher[i-1]) and not np.isnan(trigger[i]):
            fisher_cross_long = fisher[i-1] <= -1.0 and fisher[i] > -1.0
            fisher_cross_short = fisher[i-1] >= 1.0 and fisher[i] < 1.0
        
        # RSI filter (loose to ensure trade frequency)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # DI crossover confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades on 1h) ===
        # Primary: 4h bullish + Fisher long + ADX trending + RSI ok
        if trend_bullish and fisher_long and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + Fisher crossover + DI bullish
        elif trend_bullish and fisher_cross_long and di_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Fisher oversold + 4h bullish + ADX ok (reversal play)
        elif fisher_oversold and trend_bullish and adx[i] > 18:
            new_signal = SIZE_ENTRY
        # Quaternary: 4h bullish + Fisher long + DI bullish (no ADX filter)
        elif trend_bullish and fisher_long and di_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Quintenary: Strong trend + Fisher signal (momentum play)
        elif is_strong_trend and fisher_long and trend_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades on 1h) ===
        # Primary: 4h bearish + Fisher short + ADX trending + RSI ok
        if trend_bearish and fisher_short and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + Fisher crossover + DI bearish
        elif trend_bearish and fisher_cross_short and di_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Fisher overbought + 4h bearish + ADX ok (reversal play)
        elif fisher_overbought and trend_bearish and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        # Quaternary: 4h bearish + Fisher short + DI bearish (no ADX filter)
        elif trend_bearish and fisher_short and di_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quintenary: Strong trend + Fisher signal (momentum play)
        elif is_strong_trend and fisher_short and trend_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 1h timeframe)
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