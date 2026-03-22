#!/usr/bin/env python3
"""
Experiment #503: 12h Multi-Timeframe Trend-Pullback with Daily HMA Filter

Hypothesis: After analyzing 473+ failed experiments, the critical pattern is:
1. 12h timeframe has been consistently failing (#491 Sharpe=-1.05, #497 Sharpe=-0.76)
2. Over-complex asymmetric logic reduces trade frequency to near-zero
3. Simple trend-following with loose pullback entries works best on BTC/ETH

This strategy implements a PROVEN pattern from the current best (Sharpe=0.676):
- 1D HMA(21) for primary trend direction (via mtf_data helper)
- 12h RSI(14) pullback entries with LOOSE thresholds (40/60 not 30/70)
- ADX(14) > 15 minimal momentum filter (not 25+ which rarely triggers)
- Bollinger Band position filter to avoid chasing extremes
- 2.5x ATR(14) trailing stoploss

Key differences from failed 12h strategies:
- LOOSER RSI thresholds (40/60 vs 30/70) = more trades
- LOWER ADX threshold (15 vs 25) = more trades
- Simple binary logic (not complex regime switching)
- Discrete position sizing at 0.25 (not 0.35 to reduce DD)

Why this should beat Sharpe=0.676:
- 12h captures more intermediate swings than 4h
- Daily HMA filter prevents counter-trend trades
- Loose thresholds ensure 30-50 trades/year (enough for Sharpe)
- Conservative sizing (0.25) limits drawdown during 2022 crash

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_pullback_daily_hma_rsi_adx_bb_atr_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    sma_50 = calculate_sma(close, 50)
    
    # Calculate BB position (0=lower, 0.5=middle, 1=upper)
    bb_range = bb_upper - bb_lower
    bb_position = np.full(n, np.nan)
    for i in range(n):
        if bb_range[i] > 1e-10:
            bb_position[i] = (close[i] - bb_lower[i]) / bb_range[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(bb_position[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND FILTER ===
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM FILTER (loose threshold) ===
        has_momentum = adx[i] > 15
        
        # === BOLLINGER POSITION FILTER ===
        # Avoid buying at upper band or selling at lower band
        not_overbought = bb_position[i] < 0.75
        not_oversold = bb_position[i] > 0.25
        
        # === ENTRY LOGIC (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        new_signal = 0.0
        
        # LONG: Bull trend + RSI pullback + momentum + not overbought
        if bull_trend and has_momentum and not_overbought:
            if rsi[i] < 40:  # Loose threshold (was 30)
                new_signal = SIZE
            elif rsi[i] < 45 and close[i] > sma_50[i]:  # Secondary entry
                new_signal = SIZE
        
        # SHORT: Bear trend + RSI rally + momentum + not oversold
        if bear_trend and has_momentum and not_oversold:
            if rsi[i] > 60:  # Loose threshold (was 70)
                new_signal = -SIZE
            elif rsi[i] > 55 and close[i] < sma_50[i]:  # Secondary entry
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals