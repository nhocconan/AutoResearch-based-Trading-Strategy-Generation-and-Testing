#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d volume confirmation and 1w trend filter.
# Long when: price closes above upper BB(20,2) AND 6h BB width at 20-period low (squeeze) AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA(50) (bullish weekly trend).
# Short when: price closes below lower BB(20,2) AND 6h BB width at 20-period low (squeeze) AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA(50) (bearish weekly trend).
# Exit when price re-enters the Bollinger Bands (close < upper BB for longs, close > lower BB for shorts) or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture low-volatility breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring weekly trend alignment and volume confirmation, avoiding false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # normalized width
    bb_width_min = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze = bb_width <= bb_width_min  # BB width at 20-period low
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Trend filter (close > EMA50 for bullish, close < EMA50 for bearish) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_bullish = close_1w > ema_50_1w_aligned  # aligned to 6bars via align_htf_to_ltf
    weekly_bearish = close_1w < ema_50_1w_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for weekly EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(squeeze[i]) or
            np.isnan(volume_spike[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i]) or
            np.isnan(atr_6h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_squeeze = squeeze[i]
        is_weekly_bullish = weekly_bullish[i]
        is_weekly_bearish = weekly_bearish[i]
        atr_val = atr_6h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price re-enters Bollinger Bands (close below upper band)
            if price < bb_upper[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price re-enters Bollinger Bands (close above lower band)
            if price > bb_lower[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price closes above upper BB AND squeeze AND volume spike AND weekly bullish trend
            if price > bb_upper[i] and is_squeeze and vol_spike and is_weekly_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price closes below lower BB AND squeeze AND volume spike AND weekly bearish trend
            elif price < bb_lower[i] and is_squeeze and vol_spike and is_weekly_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BollingerSqueeze_1dVolumeSpike_1wTrend_V1"
timeframe = "6h"
leverage = 1.0