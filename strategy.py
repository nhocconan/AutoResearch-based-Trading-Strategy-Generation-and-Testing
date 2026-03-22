#!/usr/bin/env python3
"""
Experiment #029: 12h Trend-Follow with 1d HMA Regime + Volume Confirmation
Hypothesis: Simpler trend-following with fewer conflicting conditions will generate more trades
while maintaining quality. Key changes from failed experiments:
1. Single clear entry trigger (EMA cross + RSI filter) instead of multiple complex conditions
2. 1d HMA as primary regime filter (bullish above, bearish below)
3. Volume confirmation to avoid false breakouts
4. Cleaner stoploss logic with proper position tracking
5. Discrete position sizing (0.30 base, 0.15 half) to minimize fee churn

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper
Why this might work: Previous 12h strategies were too complex with conflicting filters.
This uses proven trend-following logic with proper regime detection.
Must generate 10+ trades on train - entry conditions deliberately loosened.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_1d_hma_vol_v3"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # HMA on 12h for faster trend
    hma_12h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h trend confirmation
        bull_trend_12h = ema_21[i] > ema_50[i]
        bear_trend_12h = ema_21[i] < ema_50[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.0
        
        # RSI filter - avoid extreme entries
        rsi_ok_long = 35 < rsi[i] < 65
        rsi_ok_short = 35 < rsi[i] < 65
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d and bull_trend_12h:
            # Primary: EMA21 > EMA50 cross with volume and RSI confirmation
            if i >= 1 and ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]:
                if vol_confirmed and rsi_ok_long and above_200:
                    new_signal = SIZE_BASE
            
            # Secondary: Pullback to EMA21 in uptrend
            elif close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99:
                if rsi[i] > 40 and rsi[i] < 55 and vol_confirmed:
                    new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d and bear_trend_12h:
            # Primary: EMA21 < EMA50 cross with volume and RSI confirmation
            if i >= 1 and ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]:
                if vol_confirmed and rsi_ok_short and below_200:
                    new_signal = -SIZE_BASE
            
            # Secondary: Bounce to EMA21 in downtrend
            elif close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01:
                if rsi[i] > 45 and rsi[i] < 60 and vol_confirmed:
                    new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Long stoploss at 2.5 ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Short stoploss at 2.5 ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                new_signal = 0.0
        
        # Update position tracking
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Position closed
            position_side = 0
            entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals