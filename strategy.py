#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d trend filter (EMA50 > EMA200).
# Long when price breaks above 1h Camarilla R3 AND 4h volume > 1.5x 20-period average AND 1d EMA50 > EMA200.
# Short when price breaks below 1h Camarilla S3 AND 4h volume > 1.5x 20-period average AND 1d EMA50 < EMA200.
# Exit when price crosses 1h Camarilla pivot point (PP) or ATR-based stop (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed to capture intraday breakouts in trending markets.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag while maintaining edge.
# Session filter: 08-20 UTC to avoid low-liquidity hours.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivot Levels (using previous day's OHLC) ===
    # For intraday, we use daily OHLC from 1d timeframe to calculate Camarilla levels for 1h
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # PP = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    rng_1d = high_1d - low_1d
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + rng_1d * 1.1 / 2.0
    s3_1d = close_1d - rng_1d * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 1h timeframe (use previous day's levels)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.5 * vol_ma_4h_aligned)
    
    # === 1d Indicators: Trend filter (EMA50 > EMA200 for long, EMA50 < EMA200 for short) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    uptrend = ema50_1d_aligned > ema200_1d_aligned
    downtrend = ema50_1d_aligned < ema200_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 200
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_1h = None
    
    # Calculate 1h ATR for stoploss
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h_raw = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, prices, atr_1h_raw)  # 1h data aligned to itself
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(uptrend[i]) or np.isnan(downtrend[i]) or np.isnan(atr_1h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_uptrend = uptrend[i]
        is_downtrend = downtrend[i]
        atr_val = atr_1h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below pivot point
            if price < pp_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above pivot point
            if price > pp_1d_aligned[i]:
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
            # LONG: Price breaks above Camarilla R3 AND volume spike AND uptrend
            if price > r3_1d_aligned[i] and vol_spike and is_uptrend:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike AND downtrend
            elif price < s3_1d_aligned[i] and vol_spike and is_downtrend:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_CamarillaR3S3_4hVolumeSpike_1dEMAFilter_V1"
timeframe = "1h"
leverage = 1.0