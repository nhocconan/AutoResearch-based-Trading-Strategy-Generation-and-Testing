#!/usr/bin/env python3
"""
Experiment #004: 1d ATR Donchian + 1w SMA Trend + Volume Spike

HYPOTHESIS: Simplify the entry to maximize signal quality:
1. Donchian(55) breakout on 1d for structure (proven in DB: Sharpe 1.10-1.38)
2. 1w SMA(50) for trend direction (filters counter-trend entries)
3. Volume spike confirmation (>2x 20-bar MA)
4. ATR(14) stoploss at 3x (gives trades room to develop)
5. Choppiness filter (CHOP < 45 = trending, avoid range markets)

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price above 1w SMA + breakout above Donchian high + volume spike = long
- Bear: Price below 1w SMA + breakdown below Donchian low + volume spike = short
- Range: CHOP > 55 = no trades (reduces whipsaw losses)
- Simple structure captures major trend changes, avoids overfitting

TARGET: 50-120 total trades over 4 years (12-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian55_sma50_vol_atr_chop"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period):
    """Donchian Channel - upper/lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - lower values = trending, higher = ranging
    CHOP < 38.2 = strong trend
    CHOP > 61.8 = range
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else 0)
            tr_sum += tr
        
        # Highest - Lowest over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_sum = highest - lowest
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w SMA(50) for trend direction ===
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=55)
    
    # Choppiness
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 150  # Donchian needs 55, ATR needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND DIRECTION ===
        htf_price = df_1w['close'].values[int(i // 4)] if i // 4 < len(df_1w) else close[i]
        htf_sma = sma_50_1w_aligned[i]
        
        bull_market = close[i] > htf_sma
        bear_market = close[i] < htf_sma
        
        # === CHOPPINESS REGIME ===
        is_trending = chop[i] < 45.0  # Trending market filter
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Bullish breakout: close crosses above upper band
        bull_breakout = prev_close <= donchian_upper[i - 1] and close[i] > donchian_upper[i]
        
        # Bearish breakdown: close crosses below lower band
        bear_breakout = prev_close >= donchian_lower[i - 1] and close[i] < donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + above 1w SMA + volume spike + trending
            if bull_breakout and bull_market and vol_spike and is_trending:
                desired_signal = SIZE
            
            # SHORT: Bearish breakdown + below 1w SMA + volume spike + trending
            elif bear_breakout and bear_market and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (3 ATR) ===
        if in_position:
            if position_side > 0:
                # Stop if price drops 3x ATR below entry
                stop_price = entry_price - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if we fall below 1w SMA (trend reversal)
                if close[i] < htf_sma:
                    desired_signal = 0.0
                
                # Exit if choppiness goes too high (range market)
                if chop[i] > 61.8:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Stop if price rises 3x ATR above entry
                stop_price = entry_price + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if we rise above 1w SMA (trend reversal)
                if close[i] > htf_sma:
                    desired_signal = 0.0
                
                # Exit if choppiness goes too high (range market)
                if chop[i] > 61.8:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals