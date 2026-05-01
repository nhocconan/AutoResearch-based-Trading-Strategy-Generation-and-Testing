#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d HMA21 trend filter and volume spike confirmation.
# Uses tighter Camarilla levels (R3/S3) for fewer, higher-quality breakouts compared to R4/S4.
# Long when price breaks above Camarilla R3 AND close > 1d HMA21 AND volume > 2.0x 20-period volume median.
# Short when price breaks below Camarilla S3 AND close < 1d HMA21 AND volume > 2.0x 20-period volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years).
# HMA21 on 1d provides smooth trend filter that works in both bull (trend continuation) and bear (sharp reversals on volume spikes at key levels).
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.

name = "4h_Camarilla_R3S3_Breakout_1dHMA21_Volume_v1"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels (using prior bar to avoid look-ahead)
    # Camarilla: R3 = close + 1.25*(high-low), S3 = close - 1.25*(high-low)
    typical_price = (high + low + close) / 3.0
    typical_price_shifted = pd.Series(typical_price).shift(1).values
    high_shifted = pd.Series(high).shift(1).values
    low_shifted = pd.Series(low).shift(1).values
    camarilla_range = high_shifted - low_shifted
    camarilla_r3 = typical_price_shifted + 1.25 * camarilla_range
    camarilla_s3 = typical_price_shifted - 1.25 * camarilla_range
    
    # Calculate 1d HMA21 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    
    # Handle arrays from convolution (they are shorter)
    diff = 2 * wma_half - wma_full
    # Pad diff to match original length
    diff_padded = np.full_like(close_1d, np.nan)
    diff_padded[half_len-1:len(diff)+half_len-1] = diff
    
    hma_21_1d = wma(diff_padded, sqrt_len)
    # Pad HMA result
    hma_21_padded = np.full_like(close_1d, np.nan)
    hma_21_padded[sqrt_len-1:len(hma_21_1d)+sqrt_len-1] = hma_21_1d
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_padded)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, HMA, volume, and Camarilla
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price vs 1d HMA21
        uptrend = curr_close > hma_21_1d_aligned[i]
        downtrend = curr_close < hma_21_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Camarilla R3 AND uptrend AND volume spike
            if curr_close > camarilla_r3[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < Camarilla S3 AND downtrend AND volume spike
            elif curr_close < camarilla_s3[i] and downtrend and volume_confirm:
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
            # Exit: price breaks below Camarilla S3 OR trend turns down
            elif curr_close < camarilla_s3[i] or not uptrend:
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
            # Exit: price breaks above Camarilla R3 OR trend turns up
            elif curr_close > camarilla_r3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals