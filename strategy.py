#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot (R3/S3) breakout with volume confirmation and 1d trend filter
# Camarilla levels from 1d capture key intraday support/resistance; volume confirms breakout validity;
# 1d EMA(34) ensures alignment with daily trend. Designed for low trade frequency (<30/year)
# to minimize fee drag in both bull and bear markets by focusing on high-probability institutional breakouts.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels: based on previous day's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng / 2
    camarilla_s3 = close_1d - 1.1 * rng / 2
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 50-period average
        vol_ma_50 = np.mean(volume[max(0, i-50):i])
        volume_spike = volume[i] > (1.8 * vol_ma_50)
        
        curr_close = close[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Camarilla R3 with 1d uptrend
                if curr_close > curr_r3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Camarilla S3 with 1d downtrend
                elif curr_close < curr_s3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Camarilla S3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla R4 level (exit at next level)
            elif curr_close >= close_1d[-1] + 1.1 * (high_1d[-1] - low_1d[-1]) if len(close_1d) > 0 else curr_close >= curr_r3 * 1.05:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Camarilla R3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla S4 level
            elif curr_close <= close_1d[-1] - 1.1 * (high_1d[-1] - low_1d[-1]) if len(close_1d) > 0 else curr_close <= curr_s3 * 0.95:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals