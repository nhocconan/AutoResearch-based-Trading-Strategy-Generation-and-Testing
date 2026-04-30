#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla R3/S3 levels with 12h trend filter and volume confirmation
# Camarilla R3/S3 act as strong daily support/resistance; breakouts with volume and 12h trend alignment
# capture institutional moves. Designed to work in both bull and bear markets by requiring
# volume confirmation and trend alignment to avoid false breakouts. Target: 50-150 total trades over 4 years.

name = "12h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels using prior 1d bar (HLC of previous day)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels based on prior 1d bar (exclude current)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # first bar has no prior
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2  # R3
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2  # S3
    camarilla_h5 = prev_close + 1.1 * (prev_high - prev_low)      # R4
    camarilla_l5 = prev_close - 1.1 * (prev_high - prev_low)      # S4
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Calculate 12h EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for dynamic position sizing and stoploss
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
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_50[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R3 with 12h uptrend
                if curr_close > camarilla_h4_aligned[i] and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 with 12h downtrend
                elif curr_close < camarilla_l4_aligned[i] and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks Camarilla S3
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla R4
            elif curr_close >= camarilla_h5_aligned[i]:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks Camarilla R3
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla S4
            elif curr_close <= camarilla_l5_aligned[i]:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals