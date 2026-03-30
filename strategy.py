#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian breakout + HMA trend + volume confirmation

HYPOTHESIS: Simple breakout strategy that works in both bull and bear markets:
- Long: Price breaks above Donchian(20) upper band + HMA(16) trending up + volume spike
- Short: Price breaks below Donchian(20) lower band + HMA(16) trending down + volume spike

WHY IT SHOULD WORK:
- Donchian channels define structural support/resistance
- HMA(16) provides smooth trend direction with less lag than SMA
- Volume spike confirms breakout legitimacy
- ATR-based stoploss handles volatility

TARGET: 75-200 total trades over 4 years (19-50/year)
TIMEFRAME: 4h (primary), 12h (HTF reference)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_12h_v3"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # Convert to pandas for rolling operations
    s = pd.Series(data)
    
    # Calculate WMA with different periods and combine
    wma1 = s.rolling(window=period, min_periods=period).apply(lambda x: np.sum(np.arange(1,len(x)+1)*x)/np.sum(np.arange(1,len(x)+1)), raw=True)
    wma2 = s.rolling(window=period//2, min_periods=period//2).apply(lambda x: np.sum(np.arange(1,len(x)+1)*x)/np.sum(np.arange(1,len(x)+1)), raw=True)
    
    # HMA = 2*WMA(period/2) - WMA(period)
    # Using approximate formula with rolling
    half_period = max(1, period // 2)
    
    # Calculate manually for better performance
    result = np.full(n, np.nan)
    for i in range(period-1, n):
        # WMA(period)
        wma_n = 0
        weight_sum = 0
        for j in range(period):
            wma_n += (period - j) * data[i - j]
            weight_sum += (period - j)
        wma_n /= weight_sum
        
        # WMA(period/2)
        wma_h = 0
        weight_sum_h = 0
        hp = min(half_period, i + 1)
        for j in range(hp):
            wma_h += (hp - j) * data[i - j]
            weight_sum_h += (hp - j)
        wma_h /= weight_sum_h
        
        result[i] = 2 * wma_h - wma_n
    
    return result

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
                adx[i] = dx
    
    adx_smooth = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # === HTF: HMA(16) on 12h for trend direction ===
    hma_12h = calculate_hma(df_12h['close'].values, 16)
    htf_trend = np.full(len(df_12h), 0.0)  # 0 = neutral, 1 = bull, -1 = bear
    
    for i in range(1, len(df_12h)):
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
            if hma_12h[i] > hma_12h[i-1]:
                htf_trend[i] = 1.0
            elif hma_12h[i] < hma_12h[i-1]:
                htf_trend[i] = -1.0
            else:
                htf_trend[i] = htf_trend[i-1]
    
    htf_trend_aligned = align_htf_to_ltf(prices, df_12h, htf_trend)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20)
    donchian_upper, donchian_middle, donchian_lower = calculate_donchian(high, low, period=20)
    
    # HMA(16) for local trend
    hma_16 = calculate_hma(close, 16)
    
    # HMA(48) for longer-term trend
    hma_48 = calculate_hma(close, 48)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Donchian 20 + HMA 48 + volume 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        
        # Donchian breakout detection
        upper_break = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        lower_break = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # HMA trend confirmation (local)
        hma_bull = hma_16[i] > hma_16[i-1] and hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_16[i-1] and hma_16[i] < hma_48[i]
        
        # HTF trend
        htf_bull = htf_trend_aligned[i] > 0.5 if not np.isnan(htf_trend_aligned[i]) else False
        htf_bear = htf_trend_aligned[i] < -0.5 if not np.isnan(htf_trend_aligned[i]) else False
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # Trend strength
        strong_trend = adx[i] > 22
        
        # === POSITION LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Upper Donchian breakout + HMA trending up + volume + HTF bull
            if upper_break and hma_bull and vol_spike:
                if htf_bull or not htf_bear:  # Allow if HTF neutral or bull
                    desired_signal = SIZE
            
            # SHORT: Lower Donchian breakout + HMA trending down + volume + HTF bear
            elif lower_break and hma_bear and vol_spike:
                if htf_bear or not htf_bull:  # Allow if HTF neutral or bear
                    desired_signal = -SIZE
        
        # === EXIT CONDITIONS ===
        if in_position:
            if position_side > 0:
                # Long exit: price falls below middle Donchian or HMA turns down
                exit_long = (close[i] < donchian_middle[i]) or (hma_16[i] < hma_16[i-1])
                
                # Stoploss: 2.5 ATR below entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                elif exit_long:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short exit: price rises above middle Donchian or HMA turns up
                exit_short = (close[i] > donchian_middle[i]) or (hma_16[i] > hma_16[i-1])
                
                # Stoploss: 2.5 ATR above entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                elif exit_short:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars ===
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