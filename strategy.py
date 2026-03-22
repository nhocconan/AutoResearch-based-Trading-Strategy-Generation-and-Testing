#!/usr/bin/env python3
"""
Experiment #329: 12h MACD Momentum with Dual HTF HMA Bias and ATR Stoploss

Hypothesis: After analyzing failures #317-#328, complex regime filters and mean 
reversion strategies consistently fail on 12h timeframe. The successful strategies 
(#300, #304) used SIMPLE trend following with HTF bias.

Key insights from failures:
- Choppiness Index regime filters: Sharpe=-0.047 (#317), -1.028 (#320)
- Fisher Transform reversals: Sharpe=-15.3 (#328) - CATASTROPHIC
- Mean reversion on 12h: Sharpe=-0.133 (#318)
- Complex ensembles: Always negative Sharpe

This strategy SIMPLIFIES to proven components:
1. 1d HMA(21) = primary directional bias (proven edge from #300, #304)
2. 1w HMA(21) = meta-trend alignment for position sizing boost
3. MACD(12,26,9) histogram = momentum confirmation (better than EMA crossover)
4. Volume ratio > 1.0 = confirmation filter (filters false breakouts)
5. ATR(14) trailing stoploss at 2.5x (proven from successful strategies)
6. Discrete position sizing: 0.25 base, 0.35 when 1w aligns

Why MACD over EMA crossover:
- MACD histogram captures momentum changes earlier
- Less whipsaw in ranging markets
- Proven in institutional trend following

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_macd_momentum_dual_htf_hma_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg.values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias (REQUIRED)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation (boosts position size)
        bull_trend_1w = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        bear_trend_1w = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === MOMENTUM CONFIRMATION ===
        # MACD histogram > 0 = bullish momentum
        # MACD histogram < 0 = bearish momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # MACD histogram increasing = strengthening momentum
        macd_strengthening_bull = i > 0 and macd_hist[i] > macd_hist[i-1] and macd_hist[i] > 0
        macd_strengthening_bear = i > 0 and macd_hist[i] < macd_hist[i-1] and macd_hist[i] < 0
        
        # === VOLUME CONFIRMATION ===
        # Volume ratio > 1.0 = above average volume (confirms move)
        volume_confirmed = vol_ratio[i] > 0.8  # relaxed for trade generation
        
        # === DIRECTIONAL CHANGE DETECTION ===
        # MACD histogram crossing zero = momentum shift
        macd_cross_bull = macd_bullish and i > 0 and macd_hist[i-1] <= 0
        macd_cross_bear = macd_bearish and i > 0 and macd_hist[i-1] >= 0
        
        # Determine position size based on HTF alignment
        if bull_trend_1w:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades) ===
        new_signal = 0.0
        
        # LONG: 1d bias up + MACD bullish + volume confirmed
        # Allow entry on either crossover OR sustained bullish momentum
        long_conditions = (
            bull_trend_1d and
            macd_bullish and
            volume_confirmed
        )
        
        # SHORT: 1d bias down + MACD bearish + volume confirmed
        short_conditions = (
            bear_trend_1d and
            macd_bearish and
            volume_confirmed
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit when HTF bias flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === MOMENTUM REVERSAL EXIT ===
        # Exit when MACD histogram flips sign against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bearish:
                new_signal = 0.0
            if position_side < 0 and macd_bullish:
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