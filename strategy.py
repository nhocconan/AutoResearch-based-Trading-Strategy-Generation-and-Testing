#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian(20) Breakout + 1w SMA(4) Trend + Volume/RSI Filter

HYPOTHESIS: Donchian(20) breakout captures institutional moves after consolidation.
Combined with 1w SMA(4) for structural trend direction and volume+RSI confirmation,
this identifies high-probability breakouts that work in both bull and bear markets.

WHY 1d: Fewer signals = less fee drag. Institutional moves are visible on daily.
WHY 1w: Weekly trend is the true structural direction filter.
WHY 1w SMA(4): ~1 month SMA, smooths weekly noise, keeps entries aligned with trend.

EXPECTED TRADES: 60-120 total over 4 years (15-30/year). HARD MAX: 150.
Win rate target: 40-50% based on proven breakout systems.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_trend_vol_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA(4) for structural trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=4, min_periods=4).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Local 1d indicators ===
    period_donchian = 20
    
    # Donchian channels (shift by 1 to avoid look-ahead)
    donchian_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().shift(1).values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # ATR(14) for stoploss sizing
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio (current vs 20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Conservative position sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = max(period_donchian + 20, 50)  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(donchian_high[i]) or np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w SMA) ===
        price_above_1w = close[i] > sma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Break above 20-bar high + volume + 1w uptrend
            if close[i] > donchian_high[i] and vol_spike and price_above_1w:
                desired_signal = SIZE
            
            # SHORT: Break below 20-bar low + volume + 1w downtrend
            if close[i] < donchian_low[i] and vol_spike and not price_above_1w:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === RSI EXIT FILTER ===
        if in_position:
            if position_side > 0 and rsi[i] > 72:
                desired_signal = 0.0
            if position_side < 0 and rsi[i] < 28:
                desired_signal = 0.0
        
        # === TRAILING STOP AFTER 2R PROFIT ===
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= 4:  # Hold at least 4 bars
                if position_side > 0:
                    profit = (highest_since_entry - entry_price) / atr[entry_bar] if atr[entry_bar] > 0 else 0
                    if profit > 2.5:
                        # Trail stop tighter
                        stop_price = highest_since_entry - 1.5 * atr[i]
                        if low[i] < stop_price:
                            desired_signal = 0.0
                
                if position_side < 0:
                    profit = (entry_price - lowest_since_entry) / atr[entry_bar] if atr[entry_bar] > 0 else 0
                    if profit > 2.5:
                        stop_price = lowest_since_entry + 1.5 * atr[i]
                        if high[i] > stop_price:
                            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        entry_bar = i
        
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * atr[i]
                else:
                    stop_price = entry_price + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals