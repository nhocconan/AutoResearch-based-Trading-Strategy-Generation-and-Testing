#!/usr/bin/env python3
"""
Experiment #380: 30m RSI Mean Reversion + 4h HMA Trend + Fisher Transform Entry + ATR Stop
Hypothesis: 30m timeframe captures intraday mean reversion opportunities while 4h HMA provides
trend bias to avoid counter-trend trades. RSI(14) extremes (35/65) with Bollinger Band(20,2)
squeeze detection identify oversold/overbought conditions. Fisher Transform(9) provides precise
entry timing on reversals. This combines proven mean reversion (RSI+BB) with trend filter (4h HMA)
and entry timing (Fisher). ATR(14) stoploss at 2.0x protects against adverse moves.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 50-150 trades total, DD < -30%.
Key insight: 30m mean reversion works better than 30m trend following (see #368, #374, #379 failures).
Position sizing: 0.25 entry, 0.15 half (take profit), discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_bb_fisher_4h_hma_trend_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate typical price
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        highest = np.max(hl2[i-period+1:i+1])
        lowest = np.min(hl2[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val == 0:
            range_val = 0.001
        
        # Normalize price to -1 to +1 range
        x = (2 * (hl2[i] - lowest) / range_val) - 1
        x = np.clip(x, -0.999, 0.999)  # Avoid division by zero in log
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (previous fisher)
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    fisher[:period] = 0.0
    trigger[:period] = 0.0
    return fisher, trigger

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX
    adx_s = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:] = adx_s
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # ADX trend strength (loose filter to allow more trades)
        is_trending = adx[i] > 15  # Very loose - most of the time
        is_weak_trend = adx[i] <= 25
        
        # RSI conditions (LOOSE for trade frequency)
        rsi_oversold = rsi[i] < 45  # Loose oversold
        rsi_overbought = rsi[i] > 55  # Loose overbought
        rsi_extreme_oversold = rsi[i] < 35
        rsi_extreme_overbought = rsi[i] > 65
        
        # Bollinger Band conditions
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        price_below_sma = close[i] < bb_sma[i]
        price_above_sma = close[i] > bb_sma[i]
        
        # Fisher Transform reversal signals
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1]
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1]
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: RSI oversold + BB lower + Fisher cross up + 4h bullish
        if rsi_oversold and price_below_lower and fisher_cross_up and trend_bullish:
            new_signal = SIZE_ENTRY
        # Secondary: RSI extreme oversold + Fisher cross up (trend neutral ok)
        elif rsi_extreme_oversold and fisher_cross_up and is_weak_trend:
            new_signal = SIZE_ENTRY
        # Tertiary: RSI oversold + price below SMA + Fisher oversold
        elif rsi_oversold and price_below_sma and fisher_oversold:
            new_signal = SIZE_ENTRY
        # Quaternary: Fisher cross up from oversold + RSI rising (ensures frequency)
        elif fisher_cross_up and fisher[i-1] < -0.5 and rsi[i] > rsi[i-1]:
            new_signal = SIZE_ENTRY
        # Fifth: Price below BB lower + RSI < 50 (simple mean reversion)
        elif price_below_lower and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: RSI overbought + BB upper + Fisher cross down + 4h bearish
        if rsi_overbought and price_above_upper and fisher_cross_down and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Secondary: RSI extreme overbought + Fisher cross down (trend neutral ok)
        elif rsi_extreme_overbought and fisher_cross_down and is_weak_trend:
            new_signal = -SIZE_ENTRY
        # Tertiary: RSI overbought + price above SMA + Fisher overbought
        elif rsi_overbought and price_above_sma and fisher_overbought:
            new_signal = -SIZE_ENTRY
        # Quaternary: Fisher cross down from overbought + RSI falling
        elif fisher_cross_down and fisher[i-1] > 0.5 and rsi[i] < rsi[i-1]:
            new_signal = -SIZE_ENTRY
        # Fifth: Price above BB upper + RSI > 50 (simple mean reversion)
        elif price_above_upper and rsi[i] > 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
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
            
            # Calculate trailing stop (2.0*ATR from lowest)
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