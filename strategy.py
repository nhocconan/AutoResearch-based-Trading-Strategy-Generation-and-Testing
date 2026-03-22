#!/usr/bin/env python3
"""
Experiment #321: 1h Regime-Adaptive Strategy with 4h HMA Bias

Hypothesis: 1h timeframe benefits from regime-adaptive logic that switches between
trend-following and mean-reversion based on market conditions.

Key components:
1. 4h HMA(21) for directional bias (proven edge from #311, #316)
2. 1h ADX(14) + Choppiness Index for regime detection
3. Trending regime (ADX>25): RSI pullback entries with HTF bias
4. Ranging regime (CHOP>61.8): RSI extreme reversals at BB bounds
5. Fisher Transform for reversal confirmation
6. ATR(14) 2.5x trailing stoploss
7. Discrete position sizing: 0.25-0.30

Why this might work on 1h:
- Pure trend following failed on 1h (#309, #315)
- Pure mean reversion failed on 1d (#312, #318)
- Regime-adaptive worked on 4h (#316 Sharpe=0.676)
- 1h has enough volatility for mean reversion but enough trend for pullbacks
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_fisher_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    high_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    low_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / (high_high - low_low + 1e-10)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50).values
    return chop

def calculate_fisher(high, low, period=9):
    """Calculate Ehlers Fisher Transform."""
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    
    normalized = 2 * (hl2 - lowest) / (highest - lowest + 1e-10) - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = fisher.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    return fisher

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for directional bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1h (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_chop(high, low, close, 14)
    fisher = calculate_fisher(high, low, 9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        trending_regime = adx[i] > 25
        ranging_regime = chop[i] > 61.8
        
        # Default to trending if neither clear
        if not trending_regime and not ranging_regime:
            if adx[i] > chop[i]:
                trending_regime = True
            else:
                ranging_regime = True
        
        # === TRENDING REGIME LOGIC (LOOSE for trade generation) ===
        # Long: 4h bias up + RSI pullback (25-55) + Fisher confirmation
        long_trend = (
            bull_trend_4h and
            trending_regime and
            25 <= rsi[i] <= 55 and
            fisher[i] > -2.0
        )
        
        # Short: 4h bias down + RSI pullback (45-75) + Fisher confirmation
        short_trend = (
            bear_trend_4h and
            trending_regime and
            45 <= rsi[i] <= 75 and
            fisher[i] < 2.0
        )
        
        # === RANGING REGIME LOGIC (LOOSE for trade generation) ===
        # Long: RSI < 35 + price near BB lower
        long_range = (
            ranging_regime and
            rsi[i] < 35 and
            close[i] <= bb_lower[i] * 1.02
        )
        
        # Short: RSI > 65 + price near BB upper
        short_range = (
            ranging_regime and
            rsi[i] > 65 and
            close[i] >= bb_upper[i] * 0.98
        )
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if long_trend or long_range:
            new_signal = SIZE_BASE
        
        if short_trend or short_range:
            new_signal = -SIZE_BASE
        
        # Boost size if strong confirmation
        if new_signal > 0 and bull_trend_4h and rsi[i] < 40:
            new_signal = SIZE_STRONG
        if new_signal < 0 and bear_trend_4h and rsi[i] > 60:
            new_signal = -SIZE_STRONG
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and ranging_regime and rsi[i] > 60:
                new_signal = 0.0
            if position_side < 0 and ranging_regime and rsi[i] < 40:
                new_signal = 0.0
        
        # === 4H BIAS REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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