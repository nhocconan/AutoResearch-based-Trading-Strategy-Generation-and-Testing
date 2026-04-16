#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Camarilla R4 (1d) AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA50 (bullish weekly trend).
# Short when price breaks below Camarilla S4 (1d) AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA50 (bearish weekly trend).
# Exit when price crosses the Camarilla pivot point (PP) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong breakouts aligned with weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current bar's high
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    PP = (prev_high + prev_low + prev_close) / 3
    R4 = PP + (prev_high - prev_low) * 1.1 / 2
    S4 = PP - (prev_high - prev_low) * 1.1 / 2
    R3 = PP + (prev_high - prev_low) * 1.1 / 4
    S3 = PP - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_bullish = close_1w > ema_50_1w_aligned  # requires align_htf_to_ltf on close_1w first
    # Fix: need to align close_1w first
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    weekly_bullish = close_1w_aligned > ema_50_1w_aligned
    weekly_bearish = close_1w_aligned < ema_50_1w_aligned
    
    # === 6h Indicators: ATR for stoploss ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    tr1_6h = pd.Series(high_6h).diff()
    tr2_6h = pd.Series(low_6h).diff().abs()
    tr3_6h = pd.Series(close_6h).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(PP_aligned[i]) or np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i]) or
            np.isnan(atr_6h_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below pivot point
            if price < PP_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above pivot point
            if price > PP_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND volume spike AND weekly bullish
            if price > R4_aligned[i] and vol_spike and weekly_bullish[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S4 AND volume spike AND weekly bearish
            elif price < S4_aligned[i] and vol_spike and weekly_bearish[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_1dVolumeSpike_1wEMA50_V1"
timeframe = "6h"
leverage = 1.0