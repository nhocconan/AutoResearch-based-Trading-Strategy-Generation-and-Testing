#!/usr/bin/env python3
"""
Experiment #226: 4h ADX Regime-Switching with Daily/Weekly HMA Trend Filter
Hypothesis: Market regime (trending vs ranging) determines which strategy works best.
ADX(14) > 25 = trending regime (use HMA crossover). ADX < 25 = ranging regime (use RSI mean reversion).
Daily HMA provides trend bias (only long when price > 1d HMA). Weekly HMA confirms macro direction.
This regime-switching approach should adapt to 2022 crash (trending down) and 2025 range market.
Position sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2.5*ATR trailing stop.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_regime_daily_weekly_hma_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_hma_crossover(hma_fast, hma_slow):
    """Detect HMA crossover signals."""
    crossover_long = (hma_fast > hma_slow) & (np.roll(hma_fast, 1) <= np.roll(hma_slow, 1))
    crossover_short = (hma_fast < hma_slow) & (np.roll(hma_fast, 1) >= np.roll(hma_slow, 1))
    crossover_long[0] = False
    crossover_short[0] = False
    return crossover_long, crossover_short

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA for trend following (fast/slow crossover)
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    hma_long_signal, hma_short_signal = calculate_hma_crossover(hma_fast, hma_slow)
    
    # Price position relative to HTF HMA
    price_above_1d = close > hma_1d_aligned
    price_below_1d = close < hma_1d_aligned
    price_above_1w = close > hma_1w_aligned
    price_below_1w = close < hma_1w_aligned
    
    # Regime detection
    trending_regime = adx > 25
    ranging_regime = adx <= 25
    
    # RSI extremes for mean reversion
    rsi_oversold = rsi < 35
    rsi_overbought = rsi > 65
    rsi_neutral = (rsi >= 35) & (rsi <= 65)
    
    # DI crossover for trend confirmation
    di_bullish = plus_di > minus_di
    di_bearish = plus_di < minus_di
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        new_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) ===
        if trending_regime[i]:
            # Long: HMA crossover + price above daily HMA + DI bullish
            if hma_long_signal[i]:
                if price_above_1d[i] and di_bullish[i]:
                    new_signal = SIZE_ENTRY
                elif price_above_1w[i] and di_bullish[i]:
                    new_signal = SIZE_ENTRY
            
            # Short: HMA crossover + price below daily HMA + DI bearish
            if hma_short_signal[i]:
                if price_below_1d[i] and di_bearish[i]:
                    new_signal = -SIZE_ENTRY
                elif price_below_1w[i] and di_bearish[i]:
                    new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (ADX <= 25) ===
        else:
            # Long: RSI oversold + price above weekly HMA (macro bullish bias)
            if rsi_oversold[i]:
                if price_above_1w[i]:
                    new_signal = SIZE_ENTRY
                elif price_above_1d[i] and rsi[i] < 30:
                    new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price below weekly HMA (macro bearish bias)
            if rsi_overbought[i]:
                if price_below_1w[i]:
                    new_signal = -SIZE_ENTRY
                elif price_below_1d[i] and rsi[i] > 70:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals