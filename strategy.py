#!/usr/bin/env python3
"""
Experiment #1380: 6h Primary + 1d/1w HTF — Donchian Breakout + RSI Pullback + Volume Confirm

Hypothesis: 6h timeframe is underexplored with ZERO successful experiments. Previous 6h failures
used overly complex regime filters (CHOP+CRSI) that generated 0 trades or negative Sharpe.

This strategy simplifies entry logic while maintaining strong HTF trend filter:
1. 1d HMA(21) + 1w HMA(21) for major trend bias (avoid counter-trend = key lesson from 2022)
2. 6h Donchian(20) breakout for entry trigger (proven on 4h/12h, untested on 6h)
3. 6h RSI(14) pullback confirmation (RSI 40-60 zone = healthy pullback, not exhaustion)
4. 6h Volume surge confirmation (taker_buy_volume ratio > 1.5 = real breakout)
5. ATR(14) trailing stop at 2.5x (mandatory risk management)

Why this should work where 6h strategies failed:
- Simpler entry = more trades (avoid 0-trade failure mode)
- Donchian breakout = catches trending moves (works in bull AND bear)
- RSI pullback filter = avoids chasing exhausted breakouts
- Volume confirm = filters false breakouts (unique vs prior 6h attempts)
- 1d+1w trend filter = prevents 2022-style crash whipsaw

Entry logic:
- LONG: price > 1d_HMA + Donchian breakout high + RSI 40-65 + volume surge
- SHORT: price < 1d_HMA + Donchian breakout low + RSI 35-60 + volume surge

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete (vol-scaled)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_rsi_pullback_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_volume_ratio(taker_buy_volume, total_volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(total_volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(total_volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    mask = vol_avg > 0
    ratio[mask] = total_volume[mask] / vol_avg[mask]
    
    return ratio

def calculate_taker_ratio(taker_buy_volume, total_volume):
    """Taker buy volume ratio (buying pressure)"""
    n = len(total_volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    mask = total_volume > 0
    ratio[mask] = taker_buy_volume[mask] / total_volume[mask]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume, period=20)
    taker_ratio = calculate_taker_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(taker_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d + 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major regime (stronger filter)
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout high = price crosses above Donchian upper
        # Breakout low = price crosses below Donchian lower
        breakout_high = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        # Volume surge = current volume > 1.5x 20-bar average
        volume_surge = vol_ratio[i] > 1.5
        
        # Taker ratio confirmation (buying/selling pressure)
        taker_buy_pressure = taker_ratio[i] > 0.55  # More buying
        taker_sell_pressure = taker_ratio[i] < 0.45  # More selling
        
        # === RSI PULLBACK ZONE ===
        # RSI 40-65 = healthy pullback in uptrend (not overbought)
        # RSI 35-60 = healthy pullback in downtrend (not oversold)
        rsi = rsi_14[i]
        rsi_long_zone = 40 <= rsi <= 65
        rsi_short_zone = 35 <= rsi <= 60
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + Donchian breakout + RSI in zone + volume confirm
        if price_above_1d and breakout_high and rsi_long_zone:
            if price_above_1w:
                # Strong trend alignment (1d + 1w both bullish)
                base_size = SIZE_STRONG
                # Volume confirm less strict in strong trend
                volume_ok = volume_surge or vol_ratio[i] > 1.2
            else:
                # Basic long (only 1d bullish)
                base_size = SIZE_BASE
                # Volume confirm required
                volume_ok = volume_surge
            
            if volume_ok and taker_buy_pressure:
                desired_signal = base_size
        
        # SHORT: 1d bearish + Donchian breakout + RSI in zone + volume confirm
        elif price_below_1d and breakout_low and rsi_short_zone:
            if price_below_1w:
                # Strong trend alignment (1d + 1w both bearish)
                base_size = SIZE_STRONG
                # Volume confirm less strict in strong trend
                volume_ok = volume_surge or vol_ratio[i] > 1.2
            else:
                # Basic short (only 1d bearish)
                base_size = SIZE_BASE
                # Volume confirm required
                volume_ok = volume_surge
            
            if volume_ok and taker_sell_pressure:
                desired_signal = -base_size
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals