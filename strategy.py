#!/usr/bin/env python3
"""
Experiment #021: 4h DEMA Cross + 12h HMA Trend + ADX Regime + Volume

HYPOTHESIS: Combine proven local signal (DEMA cross) with 12h HMA trend and ADX regime:
1. DEMA(8,21) cross - fast local signal (proven in ETH 1.32 performer)
2. 12h HMA(21) - HTF trend direction (filters false breakouts)
3. ADX(14) > 25 - trending regime (better than choppiness for trend strength)
4. Volume 1.7x - confirmation (slightly relaxed)
5. ATR(14) 2.5x trailing stop

WHY IT SHOULD WORK IN BULL AND BEAR:
- Bull: DEMA cross up + HMA up + ADX>25 = ride the rally
- Bear: DEMA cross down + HMA down + ADX>25 = short the decline
- Range (ADX<20): skip entries, avoid whipsaws
- ADX is better than CHOP because it measures TREND STRENGTH directly

KEY DIFFERENCES FROM PRIOR ATTEMPTS:
- Uses DEMA cross instead of Donchian breakout (different signal mechanism)
- Uses ADX regime instead of Choppiness Index (different regime detection)
- Previous session best used CHOP<50: this uses ADX>25 (more direct trend filter)

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dema_cross_htf_hma_adx_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_dema(prices, period=21):
    """Double Exponential Moving Average - smoother than EMA, less lag than SMA"""
    n = len(prices)
    if n < period:
        return np.full(n, np.nan)
    
    prices_s = pd.Series(prices)
    ema1 = prices_s.ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    
    dema = 2 * ema1 - ema2
    return dema.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.where(atr_smooth > 0, 100 * plus_dm_smooth / atr_smooth, 0)
    minus_di = np.where(atr_smooth > 0, 100 * minus_dm_smooth / atr_smooth, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA(21) for trend direction
    hma_21_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === Local 4h indicators ===
    dema_fast = calculate_dema(close, period=8)   # DEMA(8)
    dema_slow = calculate_dema(close, period=21)  # DEMA(21)
    adx = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 200  # DEMA(21) + ADX(14) + volume MA(20) + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(dema_fast[i]) or np.isnan(dema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or adx[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ADX REGIME FILTER ===
        strong_trend = adx[i] > 25  # Trend is strong enough to trade
        weak_trend = adx[i] < 20     # Ranging - skip
        
        # === HTF TREND (12h HMA) ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === DEMA CROSS (local signal) ===
        # Bullish: fast crosses above slow
        bullish_cross = dema_fast[i] > dema_slow[i] and dema_fast[i-1] <= dema_slow[i-1]
        # Bearish: fast crosses below slow
        bearish_cross = dema_fast[i] < dema_slow[i] and dema_fast[i-1] >= dema_slow[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.7
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Bullish cross + HTF up + strong trend + volume ===
            if bullish_cross and htf_trend_up and strong_trend and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Bearish cross + HTF down + strong trend + volume ===
            if bearish_cross and htf_trend_down and strong_trend and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if trend weakens (ADX < 20)
                if weak_trend:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if trend weakens
                if weak_trend:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals