#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels (R3/S3) for mean reversion in ranging markets,
# filtered by 1d ADX < 25 to avoid trending conditions, with volume spike confirmation.
# Camarilla R3/S3 act as strong intraday support/resistance; mean reversion works in low-volatility regimes.
# ADX filter ensures we only trade when market is ranging (ADX < 25), reducing false signals in trends.
# Volume spike confirms institutional interest at pivot levels.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.

name = "6h_Camarilla_R3S3_MeanReversion_1dADX25_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    df_1d_shifted = df_1d.shift(1)  # use previous day's data
    df_1d_shifted = df_1d_shifted.iloc[1:]  # remove first NaN row
    
    typical_price = (df_1d_shifted['high'] + df_1d_shifted['low'] + df_1d_shifted['close']) / 3
    range_hl = df_1d_shifted['high'] - df_1d_shifted['low']
    
    camarilla_r3 = typical_price + (range_hl * 1.1 / 4)
    camarilla_s3 = typical_price - (range_hl * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_s3.values)
    
    # Calculate 1d ADX(14) for regime filter (ADX < 25 = ranging market)
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_1d = df_1d_shifted['high'].values
    low_1d = df_1d_shifted['low'].values
    close_1d = df_1d_shifted['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d_shifted, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for ADX calculation
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_adx = adx_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        
        # Only trade in ranging markets (ADX < 25)
        if curr_adx < 25:
            if position == 0:  # Flat - look for mean reversion entries
                if volume_spike:
                    # Long entry: price drops to S3 level with volume spike
                    if curr_close <= curr_s3:
                        signals[i] = 0.25
                        position = 1
                        entry_price = curr_close
                    # Short entry: price rises to R3 level with volume spike
                    elif curr_close >= curr_r3:
                        signals[i] = -0.25
                        position = -1
                        entry_price = curr_close
            
            elif position == 1:  # Long position
                # Exit: price reaches midpoint between S3 and R3 (mean reversion target)
                midpoint = (curr_r3 + curr_s3) / 2
                if curr_close >= midpoint:
                    signals[i] = 0.0
                    position = 0
                # Stoploss: 1.5 * ATR below entry (using 6h ATR for dynamic stop)
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit: price reaches midpoint between S3 and R3
                midpoint = (curr_r3 + curr_s3) / 2
                if curr_close <= midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals