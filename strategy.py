#!/usr/bin/env python3
"""
Experiment #298: 4h MACD Momentum with 1d HMA Trend Bias and ADX Filter

Hypothesis: After analyzing 297 experiments, clear patterns emerge:
1. 4h timeframe with 1d HTF bias is proven edge (#292 Sharpe=0.485)
2. Simple trend following beats complex ensembles consistently
3. MACD histogram captures momentum shifts before price reversals
4. ADX filter reduces whipsaws in ranging markets
5. Faster EMA periods (8/21 vs 13/50) generate more trades on 4h

This strategy combines:
1. 1d HMA(21) slope for directional bias (proven from #292)
2. 4h MACD(12,26,9) histogram for momentum entry timing
3. ADX(14)>18 for trend strength confirmation
4. ATR(14) 2.5x trailing stoploss for risk management
5. Volume spike confirmation on breakouts (optional filter)

Why this might beat #292:
- MACD histogram leads Supertrend in momentum detection
- 4h generates more trades than 12h (#293 only Sharpe=0.111)
- Simpler than Donchian (#286 failed) while maintaining trend bias
- ADX filter prevents entries during choppy conditions

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_macd_1d_hma_adx_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator.
    Returns: macd_line, signal_line, histogram
    """
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate volume moving average for spike detection
    vol_s = pd.Series(volume)
    vol_ma = vol_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_INCREASED = 0.35  # Increased size in strong trend
    
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
        
        if np.isnan(macd_hist[i]) or np.isnan(macd_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA slope = directional bias
        # Calculate HMA slope over last 3 bars
        if i >= 3:
            hma_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        else:
            hma_slope = 0.0
        
        bull_trend_1d = hma_slope > 0
        bear_trend_1d = hma_slope < 0
        
        # === TREND STRENGTH ===
        # ADX > 18 = trending market
        trending = adx[i] > 18
        
        # === MACD MOMENTUM ===
        # MACD histogram above signal = bullish momentum building
        # MACD histogram below signal = bearish momentum building
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        
        # MACD line crossover confirmation
        macd_line_cross_up = macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1] if i > 0 else False
        macd_line_cross_down = macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        # Volume spike = 1.5x recent average
        vol_spike = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # === POSITION SIZING ===
        # Strong trend = increase size
        strong_trend = adx[i] > 30
        
        if strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up + MACD momentum + ADX trend
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1d and  # 1d HMA slope bullish
            trending and  # ADX confirms trend
            (macd_bullish or macd_line_cross_up)  # MACD momentum
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA slope bearish
            trending and  # ADX confirms trend
            (macd_bearish or macd_line_cross_down)  # MACD momentum
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === MACD REVERSAL EXIT ===
        # Exit if MACD momentum reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bearish:
                new_signal = 0.0  # MACD momentum reversed against long
            if position_side < 0 and macd_bullish:
                new_signal = 0.0  # MACD momentum reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals