#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla R3/S3 levels with 1w EMA34 trend filter and volume confirmation
# Weekly Camarilla R3/S3 act as strong support/resistance; breakouts with volume and weekly trend alignment
# capture institutional moves. Designed to work in both bull and bear markets by requiring
# volume confirmation and trend alignment to avoid false breakouts. Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w Camarilla levels using prior 1w bar (HLC of previous week)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Camarilla levels based on prior 1w bar (exclude current)
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = np.nan  # first bar has no prior
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2  # R3
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2  # S3
    camarilla_h5 = prev_close + 1.1 * (prev_high - prev_low)      # R4
    camarilla_l5 = prev_close - 1.1 * (prev_high - prev_low)      # S4
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1w bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l5)
    
    # Calculate 1w EMA(34) for trend filter
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for dynamic stoploss
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
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R3 with 1w uptrend
                if curr_close > camarilla_h4_aligned[i] and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S3 with 1w downtrend
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