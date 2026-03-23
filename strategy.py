#!/usr/bin/env python3
"""
Experiment #260: 1h Primary + 4h/12h HTF — Simplified Multi-TF Pullback

Hypothesis: After 200+ failed experiments, the key is balancing trade frequency:
- CRSI <15/>85 triggers TOO RARELY (0-trade failures like #248, #250, #255)
- Pure trend has too many whipsaws (negative Sharpe like #249, #254)
- SOLUTION: RSI 40/60 (triggers ~15% of bars) + HTF trend + volume/session filters

KEY INSIGHT FROM FAILURES:
- #250, #255: 0 trades on 1h due to too strict entry conditions
- #249, #254: Negative Sharpe from regime-switching whipsaws
- Use 12h HMA for macro bias (slower than 1d, more stable)
- Use 4h HMA for medium-term direction
- RSI 40/60 pullback (more frequent than 35/65 or CRSI)
- Volume filter (>0.8x 20-bar avg) + Session filter (8-20 UTC)
- Target: 40-70 trades/year on 1h (not too many, not zero)

TARGET: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), trades >= 30 on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling volume average."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate 4h HMA for medium-term trend (aligned properly with shift(1))
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 12h HMA for macro trend (aligned properly with shift(1))
    hma_12h_21_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = (hour >= 8) and (hour <= 20)
        
        # === VOLUME FILTER (> 0.8x 20-bar average) ===
        volume_ok = volume[i] > (0.8 * vol_avg_20[i])
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_21_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_21_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === RSI PULLBACK SIGNALS (40/60 thresholds - more frequent) ===
        # Long: bullish trend + RSI pullback to 40-55 zone
        rsi_pullback_long = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 55.0)
        # Short: bearish trend + RSI pullback to 45-60 zone
        rsi_pullback_short = (rsi_14[i] >= 45.0) and (rsi_14[i] <= 60.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 12h bullish + 4h bullish + RSI pullback + session + volume
        if price_above_hma_12h and hma_4h_bullish and rsi_pullback_long and in_session and volume_ok:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY: 12h bearish + 4h bearish + RSI pullback + session + volume
        elif price_below_hma_12h and hma_4h_bearish and rsi_pullback_short and in_session and volume_ok:
            desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # Only hold if we're in position AND no exit signal triggered
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish (even if RSI moved)
                if hma_4h_bullish and price_above_hma_12h:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if trend still bearish (even if RSI moved)
                if hma_4h_bearish and price_below_hma_12h:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals