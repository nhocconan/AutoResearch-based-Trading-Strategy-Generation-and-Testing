#!/usr/bin/env python3
"""
Experiment #038: 30m Multi-Timeframe Trend with Dual Entry Modes
Hypothesis: Combining pullback entries (RSI dip in trend) with breakout entries (Donchian) 
generates more trades while maintaining trend bias from 4h/1d HMA.
Previous 30m strategies failed due to overly strict conditions. This loosens entries
by allowing BOTH pullback AND breakout modes, increasing trade count.
Timeframe: 30m (REQUIRED), HTF: 4h and 1d via mtf_data helper.
Position sizing: 0.25-0.30 discrete, stoploss at 2.5*ATR.
Must generate 10+ trades on train, 3+ on test - dual entry modes ensure sufficient trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_dual_entry_4h_1d_hma_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel for breakout detection."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF HMAs
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend bias - 4h HMA primary
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = ema_21[i] > ema_50[i]
        bear_trend_30m = ema_21[i] < ema_50[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_pullback_long = 35 < rsi[i] < 55
        rsi_bounce_short = 45 < rsi[i] < 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        new_signal = 0.0
        
        # === LONG ENTRIES (dual mode: pullback OR breakout) ===
        if bull_trend_4h:
            # Mode 1: Pullback entry - RSI dip in uptrend
            if price_near_ema21_long and rsi_pullback_long:
                new_signal = SIZE_BASE
            
            # Mode 2: Breakout entry - Donchian breakout with trend
            elif breakout_long and bull_trend_30m:
                new_signal = SIZE_BASE
            
            # Mode 3: Oversold bounce in strong trend (4h + 1d bull)
            elif bull_trend_1d and rsi_oversold:
                new_signal = SIZE_HALF
            
            # Mode 4: Simple trend continuation
            elif bull_trend_30m and rsi[i] > 45 and rsi[i] < 65:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (dual mode: pullback OR breakout) ===
        elif bear_trend_4h:
            # Mode 1: Pullback entry - RSI bounce in downtrend
            if price_near_ema21_short and rsi_bounce_short:
                new_signal = -SIZE_BASE
            
            # Mode 2: Breakout entry - Donchian breakdown with trend
            elif breakout_short and bear_trend_30m:
                new_signal = -SIZE_BASE
            
            # Mode 3: Overbought rejection in strong trend (4h + 1d bear)
            elif bear_trend_1d and rsi_overbought:
                new_signal = -SIZE_HALF
            
            # Mode 4: Simple trend continuation
            elif bear_trend_30m and rsi[i] > 35 and rsi[i] < 55:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals