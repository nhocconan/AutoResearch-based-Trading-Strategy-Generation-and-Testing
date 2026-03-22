#!/usr/bin/env python3
"""
Experiment #036: 1d Donchian Breakout + 1w HMA Trend + Volatility Filter
Hypothesis: Daily Donchian breakouts (20-day) capture major trend moves while 1w HMA
filters counter-trend breakouts. Volatility filter (ATR ratio) avoids low-vol fakeouts.
This should generate 20-40 trades/year on daily data, enough to meet minimum trade requirements.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
Position sizing: 0.30 base, discrete levels (0.0, ±0.30) to minimize fee churn.
Key innovation: Only enter breakouts when volatility is expanding (ATR(7)/ATR(21) > 1.2)
and in direction of weekly trend. Stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_vol_filter_v1"
timeframe = "1d"
leverage = 1.0

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
    """Calculate RSI using standard formula."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

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
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    rsi = calculate_rsi(close, period=14)
    
    # EMA for additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Volatility expansion ratio
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    mask = (atr_21 != 0) & (~np.isnan(atr_7)) & (~np.isnan(atr_21))
    vol_ratio[mask] = atr_7[mask] / atr_21[mask]
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF)
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # Volatility expansion filter (avoid low-vol fakeouts)
        vol_expanding = vol_ratio[i] > 1.15  # ATR(7) > 1.15 * ATR(21)
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # RSI filter (avoid extreme entries)
        rsi_neutral = 35 < rsi[i] < 65  # Not overbought/oversold
        rsi_bullish = rsi[i] > 45  # Some bullish momentum
        rsi_bearish = rsi[i] < 55  # Some bearish momentum
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Donchian breakout + 1w bull trend + vol expanding
        if breakout_long and bull_trend and vol_expanding:
            new_signal = SIZE_BASE
        # Secondary: Donchian breakout + 1w bull trend + RSI bullish + EMA bullish
        elif breakout_long and bull_trend and rsi_bullish and ema_bullish:
            new_signal = SIZE_BASE
        # Tertiary: Donchian breakout + vol expanding + RSI neutral (weaker signal)
        elif breakout_long and vol_expanding and rsi_neutral:
            new_signal = SIZE_BASE * 0.5  # Half size for weaker setup
        
        # === SHORT ENTRY ===
        # Primary: Donchian breakout + 1w bear trend + vol expanding
        if breakout_short and bear_trend and vol_expanding:
            new_signal = -SIZE_BASE
        # Secondary: Donchian breakout + 1w bear trend + RSI bearish + EMA bearish
        elif breakout_short and bear_trend and rsi_bearish and ema_bearish:
            new_signal = -SIZE_BASE
        # Tertiary: Donchian breakout + vol expanding + RSI neutral (weaker signal)
        elif breakout_short and vol_expanding and rsi_neutral:
            new_signal = -SIZE_BASE * 0.5  # Half size for weaker setup
        
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
        
        # === EXIT ON OPPOSITE BREAKOUT ===
        # Close long on short breakout
        if position_side > 0 and breakout_short:
            new_signal = 0.0
        
        # Close short on long breakout
        if position_side < 0 and breakout_long:
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