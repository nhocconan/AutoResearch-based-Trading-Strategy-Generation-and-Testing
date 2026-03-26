#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + 1d Trend Filter

HYPOTHESIS: Price breaking a 4h Donchian channel (20 periods = ~5 days) with 
volume confirmation marks institutional moves. The 1d HMA(21) filters for trend 
direction. ATR-based stoploss manages risk. This pattern works in both bull 
(running longs) and bear (shorting breakdowns).

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks above Donchian high + 1d HMA rising = momentum confirmation
- Bear: Price breaks below Donchian low + 1d HMA falling = bearish continuation
- Volume spike confirms institutional participation, not whipsaws
- ATR stoploss prevents 2022-style crash losses

TARGET: 75-200 total trades over 4 years (~19-50/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr, 52%wr)

KEY INSIGHT: Entry conditions MUST be strict enough to generate 75-200 trades,
not the 2443 trades from #016 which was catastrophic overtrading.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_trend_v5"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr_smooth > 0, atr_smooth, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr_smooth > 0, atr_smooth, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days of 4h bars)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 4h EMA for trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (ADX for trend strength) ===
        adx = adx_14[i]
        is_trending = adx > 20  # Require some trend strength
        
        # === 1d TREND BIAS ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        hma_rising = i > 0 and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i] > hma_1d_aligned[i-1]
        hma_falling = i > 0 and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i] < hma_1d_aligned[i-1]
        
        # === DONCHIAN BREAKOUT LEVELS ===
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        
        # Previous bar's Donchian for breakout detection
        dch_high_prev = donchian_high[i-1] if i > 0 else dch_high
        dch_low_prev = donchian_low[i-1] if i > 0 else dch_low
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Strict volume requirement
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # BREAKOUT DETECTION: current close above/below previous Donchian
        # Long: close breaks above previous 4h high
        # Short: close breaks below previous 4h low
        long_breakout = close[i] > dch_high_prev and high[i] > dch_high_prev
        short_breakout = close[i] < dch_low_prev and low[i] < dch_low_prev
        
        # === STRICT ENTRY: breakout + volume + aligned trend ===
        if is_trending:
            # LONG: Breakout above + volume spike + bullish 1d trend
            if long_breakout and vol_spike:
                if price_above_1d_hma and (hma_rising or adx > 25):
                    desired_signal = SIZE
            
            # SHORT: Breakdown below + volume spike + bearish 1d trend
            if short_breakout and vol_spike:
                if not price_above_1d_hma and (hma_falling or adx > 25):
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === OPPOSITE BREAKOUT EXIT ===
        if in_position:
            if position_side > 0 and short_breakout:
                desired_signal = 0.0
            if position_side < 0 and long_breakout:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals