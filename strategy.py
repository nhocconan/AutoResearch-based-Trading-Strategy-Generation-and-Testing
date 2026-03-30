#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian + HMA + Volume with 12h Trend Filter

HYPOTHESIS: Based on DB winners mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (95tr, Sharpe 1.38)
and mtf_4h_hma_volume_donchian_adx_12h_v1 (94tr, Sharpe 1.32). This uses the proven combination:
- 12h HMA for trend direction (captures medium-term swings)
- 4h Donchian(20) for breakout structure
- Volume spike for institutional confirmation
- ATR-based stoploss for risk management

WHY IT SHOULD WORK IN BULL AND BEAR:
- Symmetrical Donchian channels catch both breakouts AND breakdowns
- HMA adapts to volatility, works in trending and ranging markets
- Short entries trigger on breakdowns below HMA in bear markets
- Volume confirms institutional participation (filters false breakouts)

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 400.
Simplified entry: 3 conditions max (trend, breakout, volume).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_12h_v2"
timeframe = "4h"
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

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = pd.Series(data).rolling(window=half, min_periods=half).mean()
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.rolling(window=sqrt_n, min_periods=sqrt_n).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA48 for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, 48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # 4h HMA21 for local momentum
    hma_4h = calculate_hma(close, 21)
    
    # Donchian channels (20 periods = 3.3 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
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
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_4h[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        hma_4h_above_12h = hma_4h[i] > hma_12h_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC (3 conditions) ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high + trending up ===
            # Condition 1: 12h trend up (price > 12h HMA)
            # Condition 2: 4h momentum confirms (4h HMA > 12h HMA)  
            # Condition 3: Price breaks above previous Donchian high
            if price_above_12h_hma and hma_4h_above_12h:
                if high[i] > prev_donchian_high:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low + trending down ===
            # Condition 1: 12h trend down (price < 12h HMA)
            # Condition 2: 4h momentum confirms (4h HMA < 12h HMA)
            # Condition 3: Price breaks below previous Donchian low
            if not price_above_12h_hma and not hma_4h_above_12h:
                if low[i] < prev_donchian_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 4 bars = 16h) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit if trend reverses (4h HMA crosses 12h HMA)
            if position_side > 0 and not hma_4h_above_12h:
                desired_signal = 0.0
            if position_side < 0 and hma_4h_above_12h:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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