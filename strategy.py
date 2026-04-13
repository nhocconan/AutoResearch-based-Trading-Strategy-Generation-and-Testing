#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Camarilla levels from daily data: fade at R3/S3 (mean reversion), breakout at R4/S4 (continuation).
# Works in bull/bear: mean reversion in range, breakout continuation in trend.
# Volume confirmation filters false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    # Using prior day's H, L, C to avoid look-ahead
    phigh = df_1d['high'].shift(1).values  # previous day high
    plow = df_1d['low'].shift(1).values    # previous day low
    pclose = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla levels: R4, R3, S3, S4
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = pclose + 1.5 * (phigh - plow)
    camarilla_r3 = pclose + 1.1 * (phigh - plow)
    camarilla_s3 = pclose - 1.1 * (phigh - plow)
    camarilla_s4 = pclose - 1.5 * (phigh - plow)
    
    # Align Camarilla levels to 6h timeframe (wait for previous day close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily trend filter: 20-period EMA
    ema_20_1d = pd.Series(pclose).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: 20-period average volume on 6s timeframe
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema20 = ema_20_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long setup: price at S3/S4 with bullish bias
            # Mean reversion long at S3/S4 when price > daily EMA
            # Breakout long above R4 when price > daily EMA
            if volume_confirm:
                if price <= s3 and price > ema20:
                    # Mean reversion long from S3/S4
                    position = 1
                    signals[i] = position_size
                elif price >= r4 and price > ema20:
                    # Breakout long above R4
                    position = 1
                    signals[i] = position_size
                elif price >= r3 and price <= r4 and price > ema20:
                    # Continuation long between R3-R4
                    position = 1
                    signals[i] = position_size * 0.5  # smaller size for continuation
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
            # Short setup: price at R3/R4 with bearish bias
            # Mean reversion short at R3/R4 when price < daily EMA
            # Breakout short below S4 when price < daily EMA
            if volume_confirm:
                if price >= r3 and price < ema20:
                    # Mean reversion short from R3/R4
                    position = -1
                    signals[i] = -position_size
                elif price <= s4 and price < ema20:
                    # Breakout short below S4
                    position = -1
                    signals[i] = -position_size
                elif price >= s3 and price <= s4 and price < ema20:
                    # Continuation short between S3-S4
                    position = -1
                    signals[i] = -position_size * 0.5  # smaller size for continuation
                else:
                    if signals[i] == 0.0:  # only set if not already set by long
                        signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R3 (take profit) or breaks below S4 (stop)
            if price >= r3 or price <= s4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S3 (take profit) or breaks above R4 (stop)
            if price <= s3 or price >= r4:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Camarilla_Pivot_Reversal_Breakout_v1"
timeframe = "6h"
leverage = 1.0