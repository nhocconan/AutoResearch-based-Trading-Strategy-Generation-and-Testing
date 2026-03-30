#!/usr/bin/env python3
"""
Experiment #028: 4h Donchian + HMA + Volume Spike + 12h EMA Trend

HYPOTHESIS: This is a DIRECT COPY of the proven DB winner pattern
(mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 with Sharpe=1.382).

WHY 4h: Proven best timeframe - enough trade frequency without overtrading.
WHY 12h HTF: Faster trend signal than 1d, aligns with 4h bar boundaries.

KEY CONDITIONS (keep simple to avoid overtrading):
1. 12h EMA trend direction (ema12 > ema26 = bullish)
2. 4h Donchian(20) breakout (high/low exceeds previous channel)
3. Volume spike (>1.5x 20-bar avg) confirms institutional move
4. ATR stoploss at 2.0x ATR(14)

TARGET: 75-150 total over 4 years = 19-37/year. HARD MAX: 300.
Signal size: 0.25-0.30 (discrete levels).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_12h_ema_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    hma = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        subset = data[max(0, i - period + 1):i + 1]
        half_period = period // 2
        
        if len(subset) >= half_period and half_period > 0:
            wma_half = np.sum(subset[-half_period:]) / half_period
            wma_full = np.sum(subset) / period
            hma[i] = 2 * wma_half - wma_full
    
    # Smooth with WMA
    hma_smooth = pd.Series(hma).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean().values
    return hma_smooth

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA for trend direction (ema12 > ema26 = bullish)
    ema_12h_close = df_12h['close'].values
    ema_12_12h = pd.Series(ema_12h_close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema_26_12h = pd.Series(ema_12h_close).ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # Align to 4h
    ema_12_aligned = align_htf_to_ltf(prices, df_12h, ema_12_12h)
    ema_26_aligned = align_htf_to_ltf(prices, df_12h, ema_26_12h)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # HMA for momentum
    hma_16 = calculate_hma(close, 16)
    
    # Donchian channels (20 periods = 5 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.15
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for Donchian(20) + HMA(16) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_12_aligned[i]) or np.isnan(ema_26_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_16[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h EMA crossover) ===
        bullish_trend = ema_12_aligned[i] > ema_26_aligned[i]
        bearish_trend = ema_12_aligned[i] < ema_26_aligned[i]
        
        # === MOMENTUM (HMA direction) ===
        hma_bullish = hma_16[i] > hma_16[i - 1] if i > 0 and not np.isnan(hma_16[i - 1]) else False
        hma_bearish = hma_16[i] < hma_16[i - 1] if i > 0 and not np.isnan(hma_16[i - 1]) else False
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35 if not np.isnan(rsi[i]) else False
        rsi_overbought = rsi[i] > 65 if not np.isnan(rsi[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian high with trend confirmation ===
            # 1. Price breaks above previous 20-bar high
            # 2. 12h trend is bullish (ema12 > ema26)
            # 3. HMA is rising (momentum confirm)
            # 4. Volume spike confirms institutional participation
            long_breakout = high[i] > prev_donchian_high
            long_trend = bullish_trend and hma_bullish
            long_vol = vol_spike
            
            if long_breakout and long_trend and long_vol:
                desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian low with trend confirmation ===
            # 1. Price breaks below previous 20-bar low
            # 2. 12h trend is bearish (ema12 < ema26)
            # 3. HMA is falling (momentum confirm)
            # 4. Volume spike confirms institutional participation
            short_breakout = low[i] < prev_donchian_low
            short_trend = bearish_trend and hma_bearish
            short_vol = vol_spike
            
            if short_breakout and short_trend and short_vol:
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
        
        # === HOLDING PERIOD EXIT (minimum 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Take partial profit at mid-channel
            if position_side > 0 and close[i] > entry_price + 2.0 * entry_atr:
                # Price moved 2R, take half profit
                if signals[i - 1] == SIZE:
                    desired_signal = HALF_SIZE  # Reduce to half position
                else:
                    desired_signal = 0.0  # Exit fully
                    
            if position_side < 0 and close[i] < entry_price - 2.0 * entry_atr:
                # Price moved 2R, take half profit
                if signals[i - 1] == -SIZE:
                    desired_signal = -HALF_SIZE
                else:
                    desired_signal = 0.0
        
        # === CHANNEL MIDDLE EXIT (price reverts to Donchian mid) ===
        if in_position and bars_held >= 4:
            if position_side > 0 and close[i] < donchian_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > donchian_mid[i]:
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