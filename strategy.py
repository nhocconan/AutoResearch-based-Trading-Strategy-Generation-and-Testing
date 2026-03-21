#!/usr/bin/env python3
"""
Experiment #039: 1h Regime-Adaptive Supertrend + 4h HMA Trend + RSI Pullback
Hypothesis: 1h timeframe balances noise reduction with trade frequency. 
4h HMA provides major trend filter. Choppiness Index detects regime (trend vs range).
In trending regime (CHOP<38.2): follow Supertrend direction with RSI pullback entries.
In ranging regime (CHOP>61.8): mean-revert with RSI extremes (long <30, short >70).
Multiple entry triggers ensure ≥10 trades per symbol. Position sizing 0.25 with 2.5x ATR stoploss.
This combines proven elements from #035 (Supertrend+CRSI) with regime adaptation for bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_supertrend_4h_hma_rsi_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = np.sum(calculate_atr(high[i-period+1:i+1], low[i-period+1:i+1], close[i-period+1:i+1], period=1))
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # 1h HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        trend_bullish_4h = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish_4h = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # Choppiness regime detection
        trending_regime = chop[i] < 45  # Trending (use trend-follow logic)
        ranging_regime = chop[i] > 55   # Ranging (use mean-reversion logic)
        
        # HMA trend on 1h
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # RSI fast for momentum
        rsi_fast_rising = rsi_fast[i] > rsi_fast[i-1] if i > 0 else True
        rsi_fast_falling = rsi_fast[i] < rsi_fast[i-1] if i > 0 else True
        
        # Bollinger Band position
        bb_lower_break = close[i] < bb_lower[i] if not np.isnan(bb_lower[i]) else False
        bb_upper_break = close[i] > bb_upper[i] if not np.isnan(bb_upper[i]) else False
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05 if not np.isnan(bb_mid[i]) else False
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price vs HMA21
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # ============ TRENDING REGIME (CHOP < 45) ============
        if trending_regime:
            # LONG entries in trending regime
            # Trigger 1: Supertrend flip long + 4h bullish
            if st_flip_long and trend_bullish_4h:
                new_signal = SIZE
            # Trigger 2: Supertrend long + RSI pullback + 4h support
            elif st_long and rsi_oversold and (trend_bullish_4h or hma_trend_long):
                new_signal = SIZE
            # Trigger 3: Supertrend long + HMA aligned + RSI rising
            elif st_long and hma_trend_long and rsi_rising and price_above_hma:
                new_signal = SIZE
            # Trigger 4: 4h bullish + Supertrend long + volume
            elif trend_bullish_4h and st_long and vol_confirm:
                new_signal = SIZE
            
            # SHORT entries in trending regime
            # Trigger 1: Supertrend flip short + 4h bearish
            if st_flip_short and trend_bearish_4h:
                new_signal = -SIZE
            # Trigger 2: Supertrend short + RSI bounce + 4h resistance
            elif st_short and rsi_overbought and (trend_bearish_4h or hma_trend_short):
                new_signal = -SIZE
            # Trigger 3: Supertrend short + HMA aligned + RSI falling
            elif st_short and hma_trend_short and rsi_falling and price_below_hma:
                new_signal = -SIZE
            # Trigger 4: 4h bearish + Supertrend short + volume
            elif trend_bearish_4h and st_short and vol_confirm:
                new_signal = -SIZE
        
        # ============ RANGING REGIME (CHOP > 55) ============
        elif ranging_regime:
            # LONG entries in ranging regime (mean reversion)
            # Trigger 1: RSI oversold + price at BB lower
            if rsi_oversold and bb_lower_break:
                new_signal = SIZE
            # Trigger 2: RSI oversold + Supertrend long (conservative)
            elif rsi_oversold and st_long:
                new_signal = SIZE
            # Trigger 3: RSI fast rising from oversold
            elif rsi[i] < 40 and rsi_fast_rising and price_above_hma:
                new_signal = SIZE
            
            # SHORT entries in ranging regime (mean reversion)
            # Trigger 1: RSI overbought + price at BB upper
            if rsi_overbought and bb_upper_break:
                new_signal = -SIZE
            # Trigger 2: RSI overbought + Supertrend short (conservative)
            elif rsi_overbought and st_short:
                new_signal = -SIZE
            # Trigger 3: RSI fast falling from overbought
            elif rsi[i] > 60 and rsi_fast_falling and price_below_hma:
                new_signal = -SIZE
        
        # ============ NEUTRAL REGIME (45 <= CHOP <= 55) ============
        else:
            # Use Supertrend direction with 4h filter
            if st_long and (trend_bullish_4h or hma_trend_long):
                new_signal = SIZE
            elif st_short and (trend_bearish_4h or hma_trend_short):
                new_signal = -SIZE
        
        # ============ STOPLOSS AND POSITION MANAGEMENT ============
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop > 0:
                    new_signal = 0.0
                elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals