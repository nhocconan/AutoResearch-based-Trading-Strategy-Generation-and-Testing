#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakouts with 1d trend filter (EMA34) and volume confirmation.
# Camarilla R1/S1 are strong intraday support/resistance levels. Breakouts with volume and 1d trend alignment
# capture high-probability moves. Designed to work in both bull and bear markets by requiring volume confirmation
# and trend alignment to avoid false breakouts. Uses 1h only for entry timing, 4h for signal direction, 1d for trend filter.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels using prior 4h bar (HLC of previous 4h bar)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla levels based on prior 4h bar (exclude current)
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = np.nan  # first bar has no prior
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels (R1, S1, R2, S2)
    camarilla_h1 = prev_close + 1.1 * (prev_high - prev_low) / 4  # R1
    camarilla_l1 = prev_close - 1.1 * (prev_high - prev_low) / 4  # S1
    camarilla_h2 = prev_close + 1.1 * (prev_high - prev_low) / 2  # R2
    camarilla_l2 = prev_close - 1.1 * (prev_high - prev_low) / 2  # S2
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l1)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l2)
    
    # Load 1d data ONCE before loop for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34) and ATR(14)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Camarilla R1 with 1d uptrend
                if curr_close > camarilla_h1_aligned[i] and curr_close > curr_ema:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Camarilla S1 with 1d downtrend
                elif curr_close < camarilla_l1_aligned[i] and curr_close < curr_ema:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Camarilla S1
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < camarilla_l1_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla R2
            elif curr_close >= camarilla_h2_aligned[i]:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Camarilla R1
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > camarilla_h1_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches Camarilla S2
            elif curr_close <= camarilla_l2_aligned[i]:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.20
    
    return signals