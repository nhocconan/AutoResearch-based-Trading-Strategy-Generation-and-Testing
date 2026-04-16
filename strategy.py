#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above upper BB AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA50 (bullish weekly trend).
# Short when price breaks below lower BB AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA50 (bearish weekly trend).
# Exit when price crosses the 4h BB midline (SMA20) or ATR stoploss (1.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture low-volatility breakouts in the direction of weekly trend.
# Target: 80-150 total trades over 4 years (20-38/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Bollinger Bands (20,2) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # BB middle (SMA20)
    bb_mid_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    # BB standard deviation
    bb_std_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper_4h = bb_mid_4h + 2 * bb_std_4h
    bb_lower_4h = bb_mid_4h - 2 * bb_std_4h
    
    # Align to 1h timeframe
    bb_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_upper_4h)
    bb_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_lower_4h)
    bb_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_mid_4h)
    
    # === 4h ATR for stoploss ===
    tr1_4h = pd.Series(high_4h).diff()
    tr2_4h = pd.Series(low_4h).diff().abs()
    tr3_4h = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Trend filter (close > EMA50 for long, close < EMA50 for short) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_bullish = close_1w > ema_50_1w_aligned
    weekly_bearish = close_1w < ema_50_1w_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bb_upper_4h_aligned[i]) or np.isnan(bb_lower_4h_aligned[i]) or np.isnan(bb_mid_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_aligned[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below BB midline
            if price < bb_mid_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above BB midline
            if price > bb_mid_4h_aligned[i]:
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
            # LONG: Price breaks above upper BB AND volume spike AND weekly bullish
            if price > bb_upper_4h_aligned[i] and vol_spike and weekly_bullish[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower BB AND volume spike AND weekly bearish
            elif price < bb_lower_4h_aligned[i] and vol_spike and weekly_bearish[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_BB20_1dVolumeSpike_1wTrend_V1"
timeframe = "4h"
leverage = 1.0