#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Weekly Camarilla R3/S3 breakout with 1d volume spike and 1w EMA(50) trend filter
# Weekly Camarilla R3/S3 levels represent strong weekly support/resistance with high reversal probability.
# Breakouts above R3 (bullish) or below S3 (bearish) with 1d volume spike indicate strong momentum.
# 1w EMA(50) filters trades to align with higher-timeframe trend, reducing false breakouts.
# Designed for low trade frequency (~12-30/year on 6h) to minimize fee drag and improve bear market performance.

name = "6h_WeeklyCamarilla_R3S3_Breakout_1wEMA50_1dVolumeSpike_v1"
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
    
    # Load 1w and 1d data ONCE before loop for Camarilla calculation, trend filter, and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels (R3, S3) using previous week's OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Camarilla formula: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_1w = close_1w + 1.1 * (high_1w - low_1w) / 2.0
    camarilla_s3_1w = close_1w - 1.1 * (high_1w - low_1w) / 2.0
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume MA(20) for volume spike confirmation
    close_1d_s = pd.Series(df_1d['close'].values)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Volume confirmation: 1d volume > 2.0x 20-period average
        vol_ma_20 = volume_ma_20_1d_aligned[i]
        volume_spike = volume[i] > (2.0 * vol_ma_20) if vol_ma_20 > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_50_1w_aligned[i]
        curr_r3 = camarilla_r3_1w_aligned[i]
        curr_s3 = camarilla_s3_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Weekly Camarilla R3 with 1w uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Weekly Camarilla S3 with 1w downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: price breaks below Weekly Camarilla S3
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Weekly Camarilla R3 (mean reversion tendency)
            elif curr_close >= curr_r3:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price breaks above Weekly Camarilla R3
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Weekly Camarilla S3 (mean reversion tendency)
            elif curr_close <= curr_s3:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals