#!/usr/bin/env python3
"""
Experiment #445: 15m Multi-Timeframe Trend Following with 4h HMA + 1h RSI Pullback + Choppiness Filter
Hypothesis: 15m timeframe needs faster HTF reference than daily. Using 4h HMA for trend bias 
provides quicker signal adaptation for 15m entries. 1h RSI pullback (not extreme) catches 
entries within the 4h trend. Choppiness Index filters out ranging markets where trend 
strategies fail. Volume confirmation ensures breakouts have participation. Multiple entry 
paths ensure >=10 trades per symbol. 2*ATR stoploss for 15m balances protection with 
allowing normal intraday volatility. Position size 0.25 keeps drawdown controlled.
Timeframe: 15m (REQUIRED), HTF: 4h and 1h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_chop_vol_atr_v1"
timeframe = "15m"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    High values (>61.8) = ranging market, Low values (<38.2) = trending market.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            trend[i] = -1
        else:
            if trend[i - 1] == 1:
                if close[i] > lower_band:
                    supertrend[i] = lower_band
                    trend[i] = 1
                else:
                    supertrend[i] = upper_band
                    trend[i] = -1
            else:
                if close[i] < upper_band:
                    supertrend[i] = upper_band
                    trend[i] = -1
                else:
                    supertrend[i] = lower_band
                    trend[i] = 1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback signals
        rsi_1h_bullish_pullback = rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60
        rsi_1h_bearish_pullback = rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60
        rsi_1h_strong_bull = rsi_1h_aligned[i] > 50
        rsi_1h_strong_bear = rsi_1h_aligned[i] < 50
        
        # 15m RSI for entry timing
        rsi_15m_oversold = rsi_15m[i] < 40
        rsi_15m_overbought = rsi_15m[i] > 60
        rsi_15m_neutral = rsi_15m[i] > 35 and rsi_15m[i] < 65
        
        # Choppiness Index regime filter
        trending_market = chop[i] < 50  # Below 50 = trending (relaxed from 38.2)
        ranging_market = chop[i] > 55   # Above 55 = ranging
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.1  # 10% above average
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Price momentum
        price_momentum_5 = (close[i] - close[i - 5]) / close[i - 5] if i > 5 else 0
        price_momentum_10 = (close[i] - close[i - 10]) / close[i - 10] if i > 10 else 0
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h HMA bullish + 1h RSI pullback + 15m RSI oversold + Trending market
        if daily_bullish and rsi_1h_bullish_pullback and rsi_15m_oversold and trending_market:
            new_signal = SIZE_ENTRY
        # Path 2: 4h HMA bullish + Supertrend bullish + Volume confirmed + RSI 35-55
        elif daily_bullish and st_bullish and volume_confirmed and rsi_15m[i] > 35 and rsi_15m[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 3: 4h HMA bullish + 1h RSI > 50 + 15m RSI crossing up from < 40
        elif daily_bullish and rsi_1h_strong_bull and rsi_15m_oversold and i > 1 and rsi_15m[i] > rsi_15m[i - 1]:
            new_signal = SIZE_ENTRY
        # Path 4: Supertrend bullish + 4h HMA bullish + Price momentum positive + Volume
        elif st_bullish and daily_bullish and price_momentum_5 > 0 and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 5: 4h HMA bullish + 15m RSI 40-55 + Trending market + Volume
        elif daily_bullish and rsi_15m[i] > 40 and rsi_15m[i] < 55 and trending_market and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 6: Supertrend bullish + 1h RSI > 45 + 15m RSI < 50 + Volume
        elif st_bullish and rsi_1h_aligned[i] > 45 and rsi_15m[i] < 50 and volume_confirmed:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: 4h HMA bearish + 1h RSI pullback + 15m RSI overbought + Trending market
        if daily_bearish and rsi_1h_bearish_pullback and rsi_15m_overbought and trending_market:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h HMA bearish + Supertrend bearish + Volume confirmed + RSI 45-65
        elif daily_bearish and st_bearish and volume_confirmed and rsi_15m[i] > 45 and rsi_15m[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h HMA bearish + 1h RSI < 50 + 15m RSI crossing down from > 60
        elif daily_bearish and rsi_1h_strong_bear and rsi_15m_overbought and i > 1 and rsi_15m[i] < rsi_15m[i - 1]:
            new_signal = -SIZE_ENTRY
        # Path 4: Supertrend bearish + 4h HMA bearish + Price momentum negative + Volume
        elif st_bearish and daily_bearish and price_momentum_5 < 0 and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 5: 4h HMA bearish + 15m RSI 45-60 + Trending market + Volume
        elif daily_bearish and rsi_15m[i] > 45 and rsi_15m[i] < 60 and trending_market and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 6: Supertrend bearish + 1h RSI < 55 + 15m RSI > 50 + Volume
        elif st_bearish and rsi_1h_aligned[i] < 55 and rsi_15m[i] > 50 and volume_confirmed:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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