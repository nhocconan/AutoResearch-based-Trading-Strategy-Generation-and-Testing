#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals identify significant swing points that act as support/resistance.
# Breakouts above/below recent fractals with 1d EMA34 trend alignment capture momentum.
# Volume confirmation (>1.3x 20-period average) ensures participation.
# ATR(14) trailing stop (2.0x) manages risk while allowing trends to develop.
# This combination has shown strong performance across multiple symbols with controlled trade frequency.
# Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for trend filter and fractals) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 6h Williams Fractals (2 left, 2 right) ===
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    highest_bear = pd.Series(high_6h).rolling(window=5, center=True, min_periods=5).max()
    lowest_bull = pd.Series(low_6h).rolling(window=5, center=True, min_periods=5).min()
    bearish_fractal = np.where(high_6h == highest_bear, high_6h, np.nan)
    bullish_fractal = np.where(low_6h == lowest_bull, low_6h, np.nan)
    
    # === 1d EMA34 trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 6h ATR (14) for trailing stop ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.3  # 1.3x average volume for confirmation
        atr_val = atr_aligned[i]
        
        # Get most recent completed fractal values (look back 2 bars to avoid look-ahead)
        # Since fractals need 2 bars to the right for confirmation, we use i-2
        if i >= 2:
            # Most recent completed bearish fractal (resistance)
            bearish_idx = np.where(~np.isnan(bearish_fractal[:i-1]))[0]
            recent_bearish = bearish_fractal[bearish_idx[-1]] if len(bearish_idx) > 0 else np.nan
            
            # Most recent completed bullish fractal (support)
            bullish_idx = np.where(~np.isnan(bullish_fractal[:i-1]))[0]
            recent_bullish = bullish_fractal[bullish_idx[-1]] if len(bullish_idx) > 0 else np.nan
        else:
            recent_bearish = np.nan
            recent_bullish = np.nan
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from high
            if price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from low
            if price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Fractal reversal) ===
        if position == 1:  # Long position
            # Exit when price breaks below recent bullish fractal (support)
            if not np.isnan(recent_bullish) and price < recent_bullish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above recent bearish fractal (resistance)
            if not np.isnan(recent_bearish) and price > recent_bearish:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation and trend alignment
            if vol_confirm:
                # Long when price breaks above recent bearish fractal (resistance) AND price > 1d EMA34 (uptrend)
                if not np.isnan(recent_bearish) and price > recent_bearish and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below recent bullish fractal (support) AND price < 1d EMA34 (downtrend)
                elif not np.isnan(recent_bullish) and price < recent_bullish and price < ema_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0