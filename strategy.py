#!/usr/bin/env python3
"""
Experiment #021: 12h ATR Channel Breakout + Weekly HMA + Volume + Choppiness

HYPOTHESIS: ATR-based channels adapt to volatility better than fixed Donchian.
- High vol → wider bands → fewer false breakouts
- Low vol → tighter bands → captures small moves
- Weekly HMA(21) for long-term trend direction
- Choppiness < 61.8 = trending (allow entries)
- Volume spike confirms institutional moves

WHY IT WORKS IN BULL AND BEAR:
- ATR channels are symmetric - work for both long and short breakouts
- Weekly trend filter prevents catching falling knives
- Bear markets: short breakouts below weekly HMA only
- Bull markets: long breakouts above weekly HMA only

TARGET: ~100 trades over 4 years (25/year). HARD MAX: 200.
Reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95tr)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_atr_channel_hma_vol_chop_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 61.8 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - price channel breakout
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Weekly Donchian for structure
    upper_1w, _, lower_1w = calculate_donchian(df_1w['high'].values, df_1w['low'].values, period=20)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # 12h Donchian for entry signals
    upper_12h, _, lower_12h = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Cooldown: minimum bars between entries (avoid overtrading)
    bars_since_exit = 999
    MIN_COOLDOWN = 12  # At least 6 days between trades (12 x 12h = 6 days)
    
    # Warmup - need 20 bars for Donchian + ATR + volume
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Update cooldown counter
        if not in_position:
            bars_since_exit += 1
        
        # === REGIME CHECK (Choppiness) ===
        chop = chop_14[i]
        is_trending = chop < 61.8  # Below 61.8 = trending (allow trades)
        
        # === WEEKLY TREND (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        weekly_bearish = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # Weekly structure: price above/below weekly channel
        price_above_1w_upper = close[i] > upper_1w_aligned[i] if not np.isnan(upper_1w_aligned[i]) else False
        price_below_1w_lower = close[i] < lower_1w_aligned[i] if not np.isnan(lower_1w_aligned[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === 12h DONCHIAN BREAKOUT ===
        upper_12 = upper_12h[i]
        lower_12 = lower_12h[i]
        
        # Check if price breaks out of 12h channel
        bullish_breakout = not np.isnan(upper_12) and close[i] > upper_12
        bearish_breakout = not np.isnan(lower_12) and close[i] < lower_12
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 12h breakout above channel + weekly bullish + trending + volume
        if not in_position and weekly_bullish and is_trending:
            # Strong entry: breakout + volume spike
            if bullish_breakout and vol_spike and bars_since_exit >= MIN_COOLDOWN:
                desired_signal = SIZE
            # Weaker entry: breakout without volume but strong weekly confirmation
            elif bullish_breakout and price_above_1w_upper and bars_since_exit >= MIN_COOLDOWN:
                desired_signal = SIZE * 0.5  # Half size without volume confirm
        
        # SHORT: 12h breakout below channel + weekly bearish + trending + volume
        if not in_position and weekly_bearish and is_trending:
            # Strong entry: breakout + volume spike
            if bearish_breakout and vol_spike and bars_since_exit >= MIN_COOLDOWN:
                desired_signal = -SIZE
            # Weaker entry: breakout without volume but strong weekly confirmation
            elif bearish_breakout and price_below_1w_lower and bars_since_exit >= MIN_COOLDOWN:
                desired_signal = -SIZE * 0.5  # Half size without volume confirm
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_exit = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        
        if in_position and desired_signal == 0.0:
            # Check if we should exit (stoploss or TP)
            in_position = False
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            bars_since_exit = 0
        
        signals[i] = desired_signal
    
    return signals