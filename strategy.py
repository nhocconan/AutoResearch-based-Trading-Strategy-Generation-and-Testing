#!/usr/bin/env python3
"""
Experiment #040: 4h KAMA Trend + Weekly HMA Regime + RSI Pullback + Volume Confirmation

Hypothesis: 4h timeframe with weekly regime filter captures multi-week trends while avoiding
intraday noise. KAMA adapts to volatility better than EMA/HMA. RSI pullback entries (40-60 range)
catch trend continuations rather than waiting for extremes. Volume confirmation reduces false
breakouts. Weekly HMA is stronger regime filter than daily (exp#030 showed daily works, weekly
should be even better for major trend direction).

Key differences from failed 4h strategies:
- Weekly HMA instead of daily (stronger regime signal, fewer whipsaws)
- KAMA instead of Supertrend (adaptively smooths in chop, responds in trends)
- RSI pullback (40-60) instead of extremes (more trades, catches continuations)
- Volume confirmation on breakouts only (not every trade = less fee churn)
- Simpler logic = fewer conflicting signals that cancel each other out

Position sizing: 0.25 base, 0.30 on high conviction (volume spike + regime aligned)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_weekly_hma_rsi_pullback_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.abs(close[:period] - close[0])
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
    volatility[:period] = change[:period]
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_21 = calculate_kama(close, 10, 2, 30)
    kama_50 = calculate_kama(close, 20, 2, 30)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    # Price momentum (ROC)
    roc_10 = np.zeros(n)
    for i in range(10, n):
        roc_10[i] = (close[i] - close[i-10]) / close[i-10] * 100 if close[i-10] > 0 else 0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    HIGH_SIZE = 0.30
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # Weekly regime filter (major trend direction)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # 4h KAMA trend
        kama_trend_long = kama_21[i] > kama_50[i] and close[i] > kama_21[i]
        kama_trend_short = kama_21[i] < kama_50[i] and close[i] < kama_21[i]
        
        # KAMA slope (momentum)
        kama_slope_long = kama_21[i] > kama_21[i-5] if i > 5 else False
        kama_slope_short = kama_21[i] < kama_21[i-5] if i > 5 else False
        
        # RSI pullback zones (not extremes - catch continuations)
        rsi_pullback_long = 40 <= rsi[i] <= 60 and rsi[i] > rsi[i-3] if i > 3 else False
        rsi_pullback_short = 40 <= rsi[i] <= 60 and rsi[i] < rsi[i-3] if i > 3 else False
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 70
        rsi_momentum_short = rsi[i] > 30 and rsi[i] < 55
        
        # Volume confirmation (spike = 1.5x average)
        vol_spike = volume[i] > vol_sma[i] * 1.5 if vol_sma[i] > 0 else False
        vol_normal = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Price momentum
        mom_long = roc_10[i] > 2
        mom_short = roc_10[i] < -2
        
        # Entry logic - multiple triggers for sufficient trades (Rule 9)
        new_signal = 0.0
        conviction = BASE_SIZE
        
        # LONG ENTRY TRIGGERS
        # Trigger 1: Weekly bullish + KAMA trend + RSI pullback (primary)
        if weekly_bullish and kama_trend_long and rsi_pullback_long:
            new_signal = BASE_SIZE
        # Trigger 2: Weekly bullish + KAMA trend + volume spike (breakout)
        elif weekly_bullish and kama_trend_long and vol_spike and mom_long:
            new_signal = HIGH_SIZE
            conviction = HIGH_SIZE
        # Trigger 3: KAMA trend + KAMA slope + RSI momentum (trend continuation)
        elif kama_trend_long and kama_slope_long and rsi_momentum_long:
            new_signal = BASE_SIZE
        # Trigger 4: Weekly bullish + RSI momentum + volume normal (steady trend)
        elif weekly_bullish and rsi_momentum_long and vol_normal and kama_slope_long:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Weekly bearish + KAMA trend + RSI pullback (primary)
        if weekly_bearish and kama_trend_short and rsi_pullback_short:
            new_signal = -BASE_SIZE
        # Trigger 2: Weekly bearish + KAMA trend + volume spike (breakdown)
        elif weekly_bearish and kama_trend_short and vol_spike and mom_short:
            new_signal = -HIGH_SIZE
            conviction = HIGH_SIZE
        # Trigger 3: KAMA trend + KAMA slope + RSI momentum (trend continuation)
        elif kama_trend_short and kama_slope_short and rsi_momentum_short:
            new_signal = -BASE_SIZE
        # Trigger 4: Weekly bearish + RSI momentum + volume normal (steady trend)
        elif weekly_bearish and rsi_momentum_short and vol_normal and kama_slope_short:
            new_signal = -BASE_SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop < entry_price:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
        
        # Update position tracking BEFORE assigning signal
        prev_signal = signals[i-1] if i > 0 else 0.0
        prev_side = np.sign(prev_signal)
        
        if new_signal != 0 and position_side == 0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                # Position flip
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            # Position closed
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals