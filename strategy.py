#!/usr/bin/env python3
"""
Experiment #399: 1h Fisher Transform + 4h HMA Trend + Volume + ATR Stop
Hypothesis: Fisher Transform normalizes price to Gaussian distribution, making extremes
clearer than RSI especially in bear/range markets (2025 test period). 4h HMA provides
trend bias (proven in current best strategy). Volume confirmation filters false breakouts.
1h timeframe is middle ground - less noisy than 15m/30m (which failed with -99% returns),
more responsive than 12h/1d. Fisher crosses at -1.5/+1.5 levels catch reversals faster.
Position size 0.25 discrete, ATR stoploss at 2.0x. Target: Beat Sharpe=0.499 baseline.
Key insight: Fisher Transform worked well in research for bear market reversals (75% win rate).
Timeframe: 1h (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_volume_atr_v1"
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

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution (-1 to +1 range typically).
    Crosses above -1.5 = long signal, crosses below +1.5 = short signal.
    Works well in bear/range markets for catching reversals.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        normalized = 0.6667 * ((close[i] - lowest) / (highest - lowest) - 0.5)
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Calculate Fisher value
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Trigger line (previous Fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_pct = np.zeros(n)
    minus_di_pct = np.zeros(n)
    mask = atr > 0
    plus_di_pct[mask] = 100 * plus_di[mask] / atr[mask]
    minus_di_pct[mask] = 100 * minus_di[mask] / atr[mask]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    di_sum = plus_di_pct + minus_di_pct
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di_pct[mask2] - minus_di_pct[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di_pct, minus_di_pct

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, trigger = calculate_fisher(close, 9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # ADX trend strength (avoid weak trends)
        is_trending = adx[i] > 20  # Relaxed from 25 to get more trades
        is_strong_trend = adx[i] > 30
        
        # Volume confirmation
        volume_ok = volume[i] > 0.8 * vol_sma[i]  # At least 80% of avg volume
        
        # Fisher Transform signals
        fisher_bullish_cross = fisher[i] > -1.5 and trigger[i] <= -1.5 if i > 0 else False
        fisher_bearish_cross = fisher[i] < 1.5 and trigger[i] >= 1.5 if i > 0 else False
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        fisher_rising = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_falling = fisher[i] < fisher[i-1] if i > 0 else False
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: Fisher bullish cross + 4h bullish + ADX trending + Volume ok
        if fisher_bullish_cross and trend_bullish and is_trending and volume_ok:
            new_signal = SIZE_ENTRY
        # Secondary: Fisher extreme long + 4h bullish + DI bullish
        elif fisher_extreme_long and trend_bullish and di_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Fisher rising + 4h bullish + Volume ok (simpler entry)
        elif fisher_rising and trend_bullish and volume_ok and fisher[i] > -1.0:
            new_signal = SIZE_ENTRY
        # Quaternary: 4h bullish + DI bullish + ADX ok (trend follow)
        elif trend_bullish and di_bullish and adx[i] > 18 and volume_ok:
            new_signal = SIZE_ENTRY
        # Quintenary: Fisher cross + DI bullish (momentum confirmation)
        elif fisher_bullish_cross and di_bullish and volume_ok:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trade frequency) ===
        # Primary: Fisher bearish cross + 4h bearish + ADX trending + Volume ok
        if fisher_bearish_cross and trend_bearish and is_trending and volume_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: Fisher extreme short + 4h bearish + DI bearish
        elif fisher_extreme_short and trend_bearish and di_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Fisher falling + 4h bearish + Volume ok (simpler entry)
        elif fisher_falling and trend_bearish and volume_ok and fisher[i] < 1.0:
            new_signal = -SIZE_ENTRY
        # Quaternary: 4h bearish + DI bearish + ADX ok (trend follow)
        elif trend_bearish and di_bearish and adx[i] > 18 and volume_ok:
            new_signal = -SIZE_ENTRY
        # Quintenary: Fisher cross + DI bearish (momentum confirmation)
        elif fisher_bearish_cross and di_bearish and volume_ok:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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