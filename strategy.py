#!/usr/bin/env python3
"""
Experiment #261: 4h Primary + 1d/1w HTF — Volume-Confirmed Trend Pullback

Hypothesis: After 250+ experiments, the key insight is:
1. Complex regime-switching (CHOP + CRSI + Donchian) creates 0-trade scenarios
2. Simple trend + pullback WORKS but needs volume confirmation to filter false signals
3. RSI zones 35-55 are too narrow — use 30/70 extremes for clearer signals
4. Volume spike (>1.5x 20-bar avg) confirms genuine momentum, not noise
5. Tighter stoploss (2.0x ATR vs 2.5x) reduces drawdown in whipsaw markets

KEY CHANGES from #251:
- Wider RSI zones (30/70 vs 35-55/45-65) for more reliable signals
- Volume confirmation (vol > 1.5x 20-bar MA) to filter fake breakouts
- Simpler hold logic — maintain position while trend intact
- Tighter stoploss (2.0x ATR) for faster exit on reversals
- 1w HMA for ultra-long-term bias (not just 1d)

TARGET: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_1d1w_atr_v1"
timeframe = "4h"
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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for ultra-long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # === RSI ENTRY SIGNALS (wider zones) ===
        # Long: RSI oversold zone with bullish setup
        rsi_oversold = rsi_14[i] <= 40.0
        # Short: RSI overbought zone with bearish setup
        rsi_overbought = rsi_14[i] >= 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1w bullish OR 1d bullish + 4h bullish + RSI oversold + volume spike
        long_condition = (
            (price_above_hma_1w or price_above_hma_1d) and
            hma_bullish and
            rsi_oversold and
            volume_spike
        )
        if long_condition:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY: 1w bearish OR 1d bearish + 4h bearish + RSI overbought + volume spike
        short_condition = (
            (price_below_hma_1w or price_below_hma_1d) and
            hma_bearish and
            rsi_overbought and
            volume_spike
        )
        if short_condition:
            desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>75)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<25)
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # Maintain position if trend still valid (reduce to half size)
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_bullish and (price_above_hma_1d or price_above_hma_1w):
                desired_signal = POSITION_SIZE_HALF
            elif position_side < 0 and hma_bearish and (price_below_hma_1d or price_below_hma_1w):
                desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    highest_since_entry = 0.0
                    lowest_since_entry = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals