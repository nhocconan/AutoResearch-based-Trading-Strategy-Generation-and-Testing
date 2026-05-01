#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 4h volume median.
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 4h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years) to minimize fee drag.
# Camarilla R3/S3 provides good balance between sensitivity and false breakout reduction.
# 1d EMA34 offers smoother trend filter than shorter periods, reducing whipsaw in choppy markets.
# Volume spike threshold set to 2.0x median to capture genuine momentum without excessive filtering.
# Added choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market) to avoid range-bound losses.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h volume median (30-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Calculate Choppiness Index (CHOP) regime filter on 4h
    # CHOP = 100 * log10(sum(ATR(14) over period) / log10(highest_high - lowest_low)) / log10(period)
    # CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(highest_high - lowest_low + 1e-10)
    chop_raw = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # avoid division by zero
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    chop_filter = chop_raw < 61.8  # Only trade in trending markets (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, volume, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_median_4h[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 4h volume median
        if vol_median_4h[i] <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 2.0)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Regime filter: only trade in trending markets
        regime_ok = chop_filter[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Camarilla R3 AND uptrend AND volume confirmation AND regime OK
            if (curr_high > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm and 
                regime_ok):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Camarilla S3 AND downtrend AND volume confirmation AND regime OK
            elif (curr_low < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm and 
                  regime_ok):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Camarilla S3 OR trend turns down OR regime changes to ranging
            elif (curr_low < camarilla_s3_aligned[i]) or (not uptrend) or (not regime_ok):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Camarilla R3 OR trend turns up OR regime changes to ranging
            elif (curr_high > camarilla_r3_aligned[i]) or (not downtrend) or (not regime_ok):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals