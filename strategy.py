#!/usr/bin/env python3
"""
Experiment #018: 1d Weekly Trend + Daily RSI Pullback Strategy
Hypothesis: Daily timeframe reduces noise and fee drag. Using 1w HMA for 
major trend bias (captures multi-month cycles), daily RSI for pullback entries,
ADX for trend strength filter, and volume confirmation. Asymmetric sizing
based on trend alignment. Conservative 0.25-0.30 position size with 2.5*ATR stop.
Timeframe: 1d (REQUIRED), HTF: 1w
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_daily_rsi_adx_vol_atr_v1"
timeframe = "1d"
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
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def calculate_momentum(close, period=10):
    """Calculate price momentum (rate of change)."""
    mom = np.zeros(len(close))
    for i in range(period, len(close)):
        mom[i] = (close[i] - close[i-period]) / close[i-period] * 100
    return mom

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    ema_1w = calculate_ema(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Moving averages
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    # Momentum
    mom_10 = calculate_momentum(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - major cycle direction
        htf_bullish = close[i] > hma_1w_aligned[i]
        htf_bearish = close[i] < hma_1w_aligned[i]
        htf_strong_bull = close[i] > ema_1w_aligned[i] and hma_1w_aligned[i] > ema_1w_aligned[i]
        htf_strong_bear = close[i] < ema_1w_aligned[i] and hma_1w_aligned[i] < ema_1w_aligned[i]
        
        # 1d trend
        price_above_sma50 = close[i] > sma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_above_ema21 = close[i] > ema_21[i]
        ema21_above_ema50 = ema_21[i] > ema_50[i]
        ema21_below_ema50 = ema_21[i] < ema_50[i]
        
        # ADX regime
        trend_strong = adx[i] > 22
        trend_weak = adx[i] < 20
        
        # RSI extremes
        rsi_oversold = rsi[i] < 38
        rsi_overbought = rsi[i] > 62
        rsi_neutral = 40 < rsi[i] < 60
        rsi_fast_oversold = rsi_fast[i] < 30
        rsi_fast_overbought = rsi_fast[i] > 70
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        vol_neutral = 0.45 < vol_ratio[i] < 0.55
        
        # Bollinger position
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        price_at_mid = bb_lower[i] * 1.02 < close[i] < bb_upper[i] * 0.98
        
        # Momentum
        mom_positive = mom_10[i] > 0
        mom_negative = mom_10[i] < 0
        mom_strong_pos = mom_10[i] > 5
        mom_strong_neg = mom_10[i] < -5
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        
        # Path 1: 1w bullish + daily pullback to EMA21 + RSI oversold (trend pullback)
        if htf_bullish and price_above_ema21 and rsi_oversold and trend_strong:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1w strong bullish + RSI fast oversold + volume bullish (dip buy)
        elif htf_strong_bull and rsi_fast_oversold and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Price > SMA200 + RSI < 40 + ADX rising (bull market dip)
        elif price_above_sma200 and rsi[i] < 40 and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # Path 4: EMA21 > EMA50 + price near BB lower + volume confirm (pullback in uptrend)
        elif ema21_above_ema50 and price_near_lower and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 5: 1w bullish + momentum positive + RSI crossing up from oversold
        if htf_bullish and mom_positive and rsi[i] > rsi[i-1] and rsi[i-1] < 35:
            new_signal = SIZE_ENTRY
        
        # Path 6: Break above EMA21 after being below + 1w bullish + volume
        elif close[i] > ema_21[i] and close[i-1] < ema_21[i-1] and htf_bullish and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        
        # Path 1: 1w bearish + daily rally to EMA21 + RSI overbought (trend pullback)
        if htf_bearish and not price_above_ema21 and rsi_overbought and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1w strong bearish + RSI fast overbought + volume bearish (rally sell)
        elif htf_strong_bear and rsi_fast_overbought and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Price < SMA200 + RSI > 60 + ADX rising (bear market rally)
        elif not price_above_sma200 and rsi[i] > 60 and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # Path 4: EMA21 < EMA50 + price near BB upper + volume confirm (rally in downtrend)
        elif ema21_below_ema50 and price_near_upper and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 1w bearish + momentum negative + RSI crossing down from overbought
        if htf_bearish and mom_negative and rsi[i] < rsi[i-1] and rsi[i-1] > 65:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Break below EMA21 after being above + 1w bearish + volume
        elif close[i] < ema_21[i] and close[i-1] > ema_21[i-1] and htf_bearish and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1d timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1d timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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