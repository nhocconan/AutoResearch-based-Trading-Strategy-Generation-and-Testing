#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian Breakout + Volume + 1d HMA Trend
HYPOTHESIS: On 12h, true Donchian(20) breakouts mark institutional moves.
Strict entry = breakout candle + volume confirmation + trend alignment.
NO position tracking = fewer trades, less complexity, less overfitting.
Target: 75-150 total trades over 4 years (19-37/year).

WHY: 12h is slow enough to avoid overtrading (vs 4h), but fast enough to
capture major trend moves. Donchian breakouts work in both bull (long upper
breakouts) and bear (short lower breakouts + rallies to HMA). Volume confirms
institutional involvement. 1d HMA filters noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_strict_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_donchian_upper(high, period=20):
    """Donchian upper band - shifted by 1 to avoid look-ahead"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    return upper

def calculate_donchian_lower(low, period=20):
    """Donchian lower band - shifted by 1 to avoid look-ahead"""
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    return lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper = calculate_donchian_upper(high, period=20)
    donch_lower = calculate_donchian_lower(low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking (simple: in_long, in_short)
    in_long = False
    in_short = False
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_long = False
            in_short = False
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_long = False
            in_short = False
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_long = False
            in_short = False
            continue
        
        # === CONDITIONS ===
        vol_spike = vol_ratio[i] > 1.5
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        rsi_val = rsi[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        upper = donch_upper[i]
        lower = donch_lower[i]
        
        # Price positions
        price_above_upper = close[i] > upper
        price_below_lower = close[i] < lower
        
        # === BREAKOUT DETECTION ===
        # True breakout: close crosses ABOVE upper band
        # Was below last bar, now above this bar
        prev_close = close[i-1] if i > warmup else close[i]
        prev_above_upper = prev_close > upper
        
        breakout_up = price_above_upper and not prev_above_upper
        
        # True breakdown: close crosses BELOW lower band
        prev_below_lower = prev_close < lower
        breakout_down = price_below_lower and not prev_below_lower
        
        desired_signal = 0.0
        
        # === NO POSITION: Look for entry ===
        if not in_long and not in_short:
            # LONG: breakout above upper + volume + bullish trend
            if breakout_up and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
                in_long = True
                in_short = False
                entry_price = close[i]
                entry_atr = atr_14[i]
                stop_price = entry_price - 2.5 * entry_atr
            
            # SHORT: breakout below lower + volume + bearish trend
            elif breakout_down and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
                in_long = False
                in_short = True
                entry_price = close[i]
                entry_atr = atr_14[i]
                stop_price = entry_price + 2.5 * entry_atr
        
        # === IN LONG: Check stoploss, exit conditions ===
        elif in_long:
            # Stoploss check
            if low[i] < stop_price:
                desired_signal = 0.0
                in_long = False
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            # Exit: opposite band break OR RSI oversold OR trend flip
            elif price_below_lower or rsi_oversold or not price_above_1d_hma:
                desired_signal = 0.0
                in_long = False
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            else:
                # Maintain position
                desired_signal = SIZE
        
        # === IN SHORT: Check stoploss, exit conditions ===
        elif in_short:
            # Stoploss check
            if high[i] > stop_price:
                desired_signal = 0.0
                in_short = False
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            # Exit: opposite band break OR RSI overbought OR trend flip
            elif price_above_upper or rsi_overbought or price_above_1d_hma:
                desired_signal = 0.0
                in_short = False
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
            else:
                # Maintain position
                desired_signal = -SIZE
        
        signals[i] = desired_signal
    
    return signals