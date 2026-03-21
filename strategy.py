#!/usr/bin/env python3
"""
Experiment #185: 12h ADX-RSI Regime Strategy with Daily/Weekly HMA Filter
Hypothesis: 12h timeframe captures multi-day swings. ADX(14) > 25 indicates trending
market (use trend-following entries), ADX < 25 indicates ranging (use mean-reversion).
Daily HMA(21) provides major trend bias, Weekly HMA(21) confirms macro direction.
RSI(14) with loosened thresholds (35/65) ensures sufficient trades. ATR(14) stoploss
at 2.5*ATR protects capital. This targets both 2022 crash (trend mode) and 2025
consolidation (range mode). Position sizing: 0.25 entry, 0.15 half-size at 2R profit.
Discrete levels minimize fees. Key improvement: ADX regime filter is more reliable
than Choppiness for distinguishing trend vs range on 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_adx_rsi_regime_daily_weekly_hma_v1"
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
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending market
    ADX < 25 = ranging market
    Reference: J. Welles Wilder
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate +DM and -DM
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm.values > minus_dm.values) & (plus_dm.values > 0), plus_dm.values, 0.0)
    minus_dm = np.where((minus_dm.values > plus_dm.values) & (minus_dm.values > 0), minus_dm.values, 0.0)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1.values, np.maximum(tr2.values, tr3.values))
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * di_diff / np.where(di_sum > 0, di_sum, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Regime detection using ADX
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] <= 25.0
        
        # 12h trend
        trend_bullish = hma_20[i] > hma_50[i] and close[i] > sma_200[i] if not np.isnan(sma_200[i]) else hma_20[i] > hma_50[i]
        trend_bearish = hma_20[i] < hma_50[i] and close[i] < sma_200[i] if not np.isnan(sma_200[i]) else hma_20[i] < hma_50[i]
        
        # RSI signals (loosened for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # MACD signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # DI crossover signals
        di_bullish = plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] if i > 0 else False
        di_bearish = minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === TREND FOLLOWING MODE (ADX > 25) ===
        if is_trending:
            # Long: DI bullish + RSI not overbought + daily/weekly bullish bias
            if di_bullish and not rsi_overbought:
                if daily_bullish or weekly_bullish:
                    new_signal = SIZE_ENTRY
            
            # Short: DI bearish + RSI not oversold + daily/weekly bearish bias
            elif di_bearish and not rsi_oversold:
                if daily_bearish or weekly_bearish:
                    new_signal = -SIZE_ENTRY
            
            # Trend continuation: HMA aligned + MACD confirmation
            elif trend_bullish and macd_hist[i] > 0 and rsi[i] > 45 and rsi[i] < 65:
                if daily_bullish:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and macd_hist[i] < 0 and rsi[i] < 55 and rsi[i] > 35:
                if daily_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === MEAN REVERSION MODE (ADX <= 25) ===
        else:
            # Long: RSI oversold + price below HMA20 + daily not strongly bearish
            if rsi_oversold and close[i] < hma_20[i]:
                if not weekly_bearish:
                    new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price above HMA20 + daily not strongly bullish
            elif rsi_overbought and close[i] > hma_20[i]:
                if not weekly_bullish:
                    new_signal = -SIZE_ENTRY
            
            # RSI divergence entry (loosened)
            elif rsi[i] < 38 and rsi_rising:
                if not daily_bearish:
                    new_signal = SIZE_ENTRY
            elif rsi[i] > 62 and rsi_falling:
                if not daily_bullish:
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