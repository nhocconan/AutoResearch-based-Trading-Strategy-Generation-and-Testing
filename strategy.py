#!/usr/bin/env python3
"""
Experiment #028: Elder Force Index Trend Riding with 1d Trend Alignment

HYPOTHESIS: Elder Force Index (EFI) = close*volume change captures institutional 
commitment better than price alone. When EFI(13) aligns with price direction, 
institutional money is behind the move. Entry on pullbacks (RSI extremes) within 
confirmed trends catches rides without needing breakout precision. 12h timeframe 
reduces noise while staying liquid. Works in bull (long pullbacks) and bear 
(short rallies) because we only enter in the direction of confirmed trends.

WHY 1d HTF: Larger timeframe trend filter prevents fighting major trends.
12h is slow enough for institutional moves but fast enough for 50-100 trades/year.

TARGET: 60-150 total trades over 4 years (15-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_efi_force_1d_ema_v1"
timeframe = "12h"
leverage = 1.0

def calculate_force_index(close, volume, period=13):
    """Elder Force Index - measures institutional pressure"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    # Force = (close - prev_close) * volume
    close_change = np.diff(close, prepend=np.nan)
    close_change[0] = 0.0
    force = close_change * volume
    
    # EMA of force
    efi = pd.Series(force).ewm(span=period, min_periods=period, adjust=False).mean().values
    return efi

def calculate_rsi(close, period=14):
    """RSI with min_periods"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

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

def calculate_ema(data, period, min_periods=None):
    """EMA with proper min_periods"""
    if min_periods is None:
        min_periods = period
    return pd.Series(data).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend direction
    ema_1d = calculate_ema(df_1d['close'].values, period=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Elder Force Index (13-period) + EMA(2) for signal
    force_raw = calculate_force_index(close, volume, period=13)
    force_ema2 = calculate_ema(force_raw, period=2, min_periods=2)
    force_ema13 = calculate_ema(force_raw, period=13, min_periods=13)
    
    # Volume confirmation
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(force_ema2[i]) or np.isnan(force_ema13[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        trend_bullish = price_above_1d_ema
        trend_bearish = not price_above_1d_ema
        
        # === FORCE INDEX DIRECTION ===
        # EMA(2) crossing EMA(13) = momentum shift
        force_positive = force_ema2[i] > 0
        force_negative = force_ema2[i] < 0
        
        # === RSI FOR PULLBACK ENTRY ===
        rsi_val = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.0
        
        # === EFI TREND CONFIRMATION ===
        # When EMA(2) > EMA(13), institutional pressure is positive
        efi_trend_up = force_ema2[i] > force_ema13[i]
        efi_trend_down = force_ema2[i] < force_ema13[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # EFI trending up + price above 1d EMA + RSI pulling back (oversold)
            if efi_trend_up and trend_bullish:
                # Pullback entry: RSI < 45 (not extremely oversold, just a pullback)
                if rsi_val < 45:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # EFI trending down + price below 1d EMA + RSI pulling back (overbought)
            if efi_trend_down and trend_bearish:
                # Rally entry: RSI > 55 (not extremely overbought, just a rally)
                if rsi_val > 55:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: Opposite trend or RSI extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: trend flips bearish OR RSI reaches overbought
            if trend_bearish:
                exit_triggered = True
            if rsi_val > 75:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: trend flips bullish OR RSI reaches oversold
            if trend_bullish:
                exit_triggered = True
            if rsi_val < 25:
                exit_triggered = True
        
        if exit_triggered:
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
                # Same direction - maintain position
                pass
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