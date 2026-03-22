#!/usr/bin/env python3
"""
Experiment #214: 4h DEMA Crossover + 1d/1w HMA Confluence + Volume + ATR Stop

Hypothesis: 4h timeframe captures multi-day swings with less noise than 15m/1h.
DEMA (Double EMA) reacts faster than regular EMA, catching trend changes earlier.
Using BOTH 1d AND 1w HMA for confluence creates stronger HTF bias filter.
Volume ratio confirms breakout validity (avoid fake breakouts on low volume).
ATR trailing stop protects against reversals.

Why this might beat current best (mtf_4h_kama_1d_hma_adx_atr_v1, Sharpe=0.478):
- DEMA vs KAMA: DEMA has less lag, catches moves 1-2 bars earlier
- Dual HTF filter (1d + 1w): Stronger trend confluence than single 1d HMA
- Volume confirmation: Filters false breakouts that hurt pure price strategies
- ADX > 15 (not 20+): Lower threshold ensures enough trades on 4h timeframe
- Conservative sizing (0.30): Controls DD in 2022-style crashes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dema_1d_1w_hma_vol_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_dema(close, period=21):
    """Calculate Double Exponential Moving Average (less lag than EMA)."""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    dema = 2 * ema1 - ema2
    return dema.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.ewm(span=period, min_periods=period, adjust=False).mean()
    vol_ratio = volume / (vol_ma.values + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    dema_fast = calculate_dema(close, 8)
    dema_slow = calculate_dema(close, 21)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(dema_fast[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # Both 1d AND 1w HMA must agree for strong confluence
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both 1d and 1w agree
        strong_bull_trend = bull_trend_1d and bull_trend_1w
        # Strong bearish: both 1d and 1w agree
        strong_bear_trend = bear_trend_1d and bear_trend_1w
        # Weaker signal: only 1d agrees (still tradable but smaller size)
        weak_bull_trend = bull_trend_1d and not bull_trend_1w
        weak_bear_trend = bear_trend_1d and not bear_trend_1w
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold for 4h to ensure trades)
        trend_strength = adx[i] > 15
        
        # === DEMA CROSSOVER ===
        # Fast DEMA crosses above slow DEMA = bullish
        # Fast DEMA crosses below slow DEMA = bearish
        dema_bullish = dema_fast[i] > dema_slow[i]
        dema_bearish = dema_fast[i] < dema_slow[i]
        
        # Check for actual crossover (not just above/below)
        dema_cross_long = dema_fast[i] > dema_slow[i] and dema_fast[i-1] <= dema_slow[i-1]
        dema_cross_short = dema_fast[i] < dema_slow[i] and dema_fast[i-1] >= dema_slow[i-1]
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.2 = above average volume (confirms breakout)
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === RSI MOMENTUM ===
        # RSI > 45 = not oversold (for longs)
        # RSI < 55 = not overbought (for shorts)
        rsi_ok_long = rsi[i] > 45
        rsi_ok_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: Strong HTF trend OR (weak HTF + crossover) + ADX + volume + RSI
        if strong_bull_trend and trend_strength and dema_bullish:
            if volume_confirmed and rsi_ok_long:
                new_signal = SIZE_BASE
            elif dema_cross_long and rsi_ok_long:
                # Crossover entry even without volume spike
                new_signal = SIZE_BASE * 0.7  # Smaller size on crossover alone
        
        if weak_bull_trend and trend_strength and dema_cross_long:
            if rsi_ok_long:
                new_signal = SIZE_BASE * 0.5  # Even smaller on weak trend
        
        # Short: Strong HTF trend OR (weak HTF + crossover) + ADX + volume + RSI
        if strong_bear_trend and trend_strength and dema_bearish:
            if volume_confirmed and rsi_ok_short:
                new_signal = -SIZE_BASE
            elif dema_cross_short and rsi_ok_short:
                new_signal = -SIZE_BASE * 0.7
        
        if weak_bear_trend and trend_strength and dema_cross_short:
            if rsi_ok_short:
                new_signal = -SIZE_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position:
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
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals