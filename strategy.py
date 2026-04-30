#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla R3/S3 breakout with 1w EMA(50) trend filter and volume spike confirmation
# Daily Camarilla levels provide intraday support/resistance with institutional relevance.
# The 1w EMA(50) ensures trades align with longer-term trend, reducing whipsaw in ranging markets.
# Volume spike confirms participation. Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion at extreme levels.

name = "12h_DailyCamarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on prior day's OHLC)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = typical_price + hl_range * 1.1 / 4
    camarilla_s3 = typical_price - hl_range * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe (wait for daily bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (1.5 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above daily R3 with 1w uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below daily S3 with 1w downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks daily S3 (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches daily R4 (mean reversion tendency)
            # R4 = C + (H-L)*1.1/2 = R3 + (H-L)*1.1/4
            hl_range_1d = (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_r4 = typical_price_1d + hl_range_1d * 1.1 / 2
            camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_r4))[i] if len(df_1d) > 0 else curr_r3
            if curr_close >= camarilla_r4_aligned:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks daily R3 (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches daily S4 (mean reversion tendency)
            # S4 = C - (H-L)*1.1/2 = S3 - (H-L)*1.1/4
            hl_range_1d = (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) if len(df_1d) > 0 else 0
            typical_price_1d = (df_1d['high'].iloc[-1] + df_1d['low'].iloc[-1] + df_1d['close'].iloc[-1]) / 3 if len(df_1d) > 0 else 0
            camarilla_s4 = typical_price_1d - hl_range_1d * 1.1 / 2
            camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(df_1d['close'].values, camarilla_s4))[i] if len(df_1d) > 0 else curr_s3
            if curr_close <= camarilla_s4_aligned:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals