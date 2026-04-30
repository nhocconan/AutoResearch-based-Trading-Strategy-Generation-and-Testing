#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 breakout with 1d EMA(50) trend filter and volume spike confirmation
# 12h Camarilla levels provide key intraday support/resistance with institutional relevance.
# The 1d EMA(50) ensures trades align with medium-term trend, reducing whipsaw in ranging markets.
# Volume spike confirms participation. Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion at extreme levels.
# Uses 1d HTF for EMA calculation as specified in experiment #110799.

name = "6h_12hCamarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (based on prior 12h bar's OHLC)
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    hl_range = df_12h['high'] - df_12h['low']
    camarilla_r3 = typical_price + hl_range * 1.1 / 4
    camarilla_s3 = typical_price - hl_range * 1.1 / 4
    
    # Align 12h Camarilla levels to 6h timeframe (wait for 12h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    
    # Load daily data ONCE before loop for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d_s = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 12h R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 12h S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 12h S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 12h R4 (mean reversion tendency)
            # R4 = C + (H-L)*1.1/2 = R3 + (H-L)*1.1/4
            hl_range_12h = (df_12h['high'].iloc[-1] - df_12h['low'].iloc[-1]) if len(df_12h) > 0 else 0
            typical_price_12h = (df_12h['high'].iloc[-1] + df_12h['low'].iloc[-1] + df_12h['close'].iloc[-1]) / 3 if len(df_12h) > 0 else 0
            camarilla_r4 = typical_price_12h + hl_range_12h * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, np.full_like(df_12h['close'].values, camarilla_r4))[i] if len(df_12h) > 0 else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 12h R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 12h S4 (mean reversion tendency)
            # S4 = C - (H-L)*1.1/2 = S3 - (H-L)*1.1/4
            hl_range_12h = (df_12h['high'].iloc[-1] - df_12h['low'].iloc[-1]) if len(df_12h) > 0 else 0
            typical_price_12h = (df_12h['high'].iloc[-1] + df_12h['low'].iloc[-1] + df_12h['close'].iloc[-1]) / 3 if len(df_12h) > 0 else 0
            camarilla_s4 = typical_price_12h - hl_range_12h * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, np.full_like(df_12h['close'].values, camarilla_s4))[i] if len(df_12h) > 0 else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals