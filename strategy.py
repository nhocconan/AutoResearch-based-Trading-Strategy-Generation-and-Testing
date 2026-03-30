#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + RSI Momentum + 1w Trend + Volume

HYPOTHESIS: The combination of Donchian channel breakout (price structure) +
RSI momentum confirmation (filters false breakouts) + 1w trend direction
(aligns with higher timeframe) + volume confirmation (validates institutional
participation) creates a robust, generalizable system.

WHY 12h: Slower than 4h (reduces fee drag), faster than 1d (captures more
opportunities). 12h Donchian(20) = 10-day channel captures medium-term swings.

WHY IT WORKS IN BULL AND BEAR: Uses symmetrical Donchian channels. Long breakouts
work in bull markets when price clears resistance. Short breakouts work in bear
when price breaks down with volume. RSI filter prevents chasing weak moves.
Weekly trend keeps us on the right side of major market direction.

TARGET: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_vol_1w_v1"
timeframe = "12h"
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

def calculate_rsi(prices, period=14):
    """RSI with proper min_periods"""
    close_s = pd.Series(prices)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for strong trend direction (smoother than SMA)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(250, donchian_period + 50)  # Need enough for Donchian(20) + EMA200(1w)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === WEEKLY TREND DIRECTION ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PREVIOUS BAR VALUES ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # === RSI MOMENTUM ===
        rsi_val = rsi_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high with confirmation ===
            # Price CLOSES above previous 20-bar high (not just intrabar spike)
            if prev_close <= prev_donchian_high and close[i] > prev_donchian_high:
                # Trend aligned: above weekly EMA
                if price_above_1w_ema:
                    # Momentum: RSI > 55 (not overbought but has strength)
                    if rsi_val > 55:
                        # Volume confirms institutional participation
                        if vol_spike:
                            desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low with confirmation ===
            # Price CLOSES below previous 20-bar low
            if prev_close >= prev_donchian_low and close[i] < prev_donchian_low:
                # Trend aligned: below weekly EMA
                if not price_above_1w_ema:
                    # Momentum: RSI < 45 (not oversold but has weakness)
                    if rsi_val < 45:
                        # Volume confirms institutional participation
                        if vol_spike:
                            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR from entry) ===
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
        
        # === MID-CHANNEL EXIT (mean reversion at 2R profit) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:  # Hold at least 2 bars
            # Check if we have 2R profit
            if position_side > 0:
                profit_r = (close[i] - entry_price) / entry_atr
                if profit_r >= 2.0:
                    # Exit when price crosses back to mid-channel
                    if close[i] < donchian_mid[i]:
                        desired_signal = 0.0
            
            if position_side < 0:
                profit_r = (entry_price - close[i]) / entry_atr
                if profit_r >= 2.0:
                    # Exit when price crosses back to mid-channel
                    if close[i] > donchian_mid[i]:
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