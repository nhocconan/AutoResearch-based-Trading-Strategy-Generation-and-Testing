#!/usr/bin/env python3
"""
Experiment #035: 12h Volatility Mean Reversion with 1d Trend Regime
Hypothesis: 12h timeframe captures volatility cycles better than lower TFs. 
Combine 1d HMA for regime filter + 12h Bollinger %B for mean reversion entries + ATR ratio for vol spike confirmation.
Key insight: Failed experiments used too many filters (0 trades) or pure trend following (whipsaw in 2022).
This strategy enters on vol spikes + extreme BB readings WITH trend direction (asymmetric).
Position sizing: 0.25-0.35 discrete levels, stoploss at 2.5*ATR.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Why this might work: Vol spike mean reversion has 75%+ win rate, 1d HMA filters false signals in chop.
Must generate 10+ trades on train - BB %B extremes happen frequently enough.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_meanrev_1d_hma_bb_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and %B."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # %B = (close - lower) / (upper - lower)
    bb_range = upper - lower
    pct_b = np.zeros(len(close))
    mask = bb_range > 0
    pct_b[mask] = (close[mask] - lower[mask]) / bb_range[mask]
    pct_b[~mask] = 0.5
    return upper, lower, pct_b, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_pct_b, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # EMA for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_pct_b[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # Volatility spike detection (ATR ratio > 1.5 = elevated vol)
        vol_spike = atr_ratio[i] > 1.5
        vol_normal = atr_ratio[i] < 1.2
        
        # Bollinger %B extremes (mean reversion signals)
        bb_oversold = bb_pct_b[i] < 0.15  # Near lower band
        bb_overbought = bb_pct_b[i] > 0.85  # Near upper band
        bb_extreme = bb_oversold or bb_overbought
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        
        # Trend confirmation on 12h
        trend_up = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        trend_down = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (bull regime + oversold conditions) ===
        if bull_regime:
            # Primary: BB oversold + vol spike + RSI confirmation
            if bb_oversold and vol_spike and rsi_oversold:
                new_signal = SIZE_BASE
            
            # Secondary: BB oversold + trend up (pullback entry)
            elif bb_oversold and trend_up and rsi[i] < 50:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in bull regime
            elif rsi_oversold and bull_regime and above_200:
                new_signal = SIZE_HALF
            
            # Momentum: Price near EMA21 pullback in uptrend
            elif close[i] <= ema_21[i] * 1.01 and trend_up and rsi[i] > 40:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (bear regime + overbought conditions) ===
        elif bear_regime:
            # Primary: BB overbought + vol spike + RSI confirmation
            if bb_overbought and vol_spike and rsi_overbought:
                new_signal = -SIZE_BASE
            
            # Secondary: BB overbought + trend down (bounce entry)
            elif bb_overbought and trend_down and rsi[i] > 50:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in bear regime
            elif rsi_overbought and bear_regime and below_200:
                new_signal = -SIZE_HALF
            
            # Momentum: Price near EMA21 bounce in downtrend
            elif close[i] >= ema_21[i] * 0.99 and trend_down and rsi[i] < 60:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
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