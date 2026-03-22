#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian Breakout + 1d HMA Trend + ATR Stop
Hypothesis: Donchian breakouts (20-period) generate sufficient trades on 12h timeframe
while 1d HMA provides trend filter to avoid counter-trend trades. ATR trailing stop
at 2.5*ATR limits drawdown. This should produce 20-50 trades/year with positive Sharpe.

Key innovations vs failed strategies:
- LOOSER entry conditions (Donchian breakout vs RSI extremes) to ensure >=10 trades
- 1d HMA trend filter (not 4h) for cleaner signal on 12h timeframe
- Simple logic: breakout + trend alignment = entry, ATR breach = exit
- Discrete position sizing (0.0, ±0.25, ±0.30) to minimize fee churn

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gains = np.zeros(n - 1)
    losses = np.zeros(n - 1)
    gains[delta > 0] = delta[delta > 0]
    losses[delta < 0] = -delta[delta < 0]
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i - 1] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i - 1] / avg_loss[i - 1]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate TR, +DM, -DM
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[1:] = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm[1:] = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr * 100
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
    
    adx[period * 2 - 1:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period * 2 - 1:]
    
    return adx

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
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # ADX trend strength (avoid trading in choppy markets)
        trend_strong = adx[i] > 20  # ADX > 20 = trending
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # RSI filter (avoid extreme overbought/oversold entries)
        rsi_ok_long = rsi[i] < 70  # Not extremely overbought
        rsi_ok_short = rsi[i] > 30  # Not extremely oversold
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Donchian breakout + 1d bull trend + ADX strong
        if breakout_long and bull_trend and trend_strong and rsi_ok_long:
            new_signal = SIZE_MAX
        # Secondary: Donchian breakout + EMA bullish confirmation
        elif breakout_long and ema_bullish and rsi_ok_long:
            new_signal = SIZE_BASE
        # Tertiary: Price above 1d HMA + RSI crossing up from oversold
        elif bull_trend and rsi[i] > 40 and rsi[i - 1] < 40:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Donchian breakout + 1d bear trend + ADX strong
        if breakout_short and bear_trend and trend_strong and rsi_ok_short:
            new_signal = -SIZE_MAX
        # Secondary: Donchian breakout + EMA bearish confirmation
        elif breakout_short and ema_bearish and rsi_ok_short:
            new_signal = -SIZE_BASE
        # Tertiary: Price below 1d HMA + RSI crossing down from overbought
        elif bear_trend and rsi[i] < 60 and rsi[i - 1] > 60:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals