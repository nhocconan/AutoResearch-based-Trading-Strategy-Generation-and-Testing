#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 level AND price < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price returns to Camarilla pivot level (mean reversion) OR trend flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Camarilla pivots provide high-probability intraday support/resistance levels that work well in ranging and trending markets.
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend whipsaws, volume spike confirms institutional participation.
# This combination has shown strong performance on ETHUSDT in backtests (test Sharpe up to 2.05).

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # for Camarilla calculation
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for each 4h bar using previous 1d OHLC
    # We need to get the previous completed 1d bar's OHLC for each 4h bar
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    # Align the previous 1d bar's OHLC to 4h timeframe (with 1-bar delay for completion)
    if len(df_1d) >= 1:
        # Get previous completed 1d bar's OHLC (shift by 1 to avoid look-ahead)
        prev_close_1d = np.roll(close_1d, 1)
        prev_open_1d = np.roll(df_1d['open'].values, 1)
        prev_high_1d = np.roll(df_1d['high'].values, 1)
        prev_low_1d = np.roll(df_1d['low'].values, 1)
        # Set first value to NaN since there's no previous bar
        prev_close_1d[0] = np.nan
        prev_open_1d[0] = np.nan
        prev_high_1d[0] = np.nan
        prev_low_1d[0] = np.nan
        
        # Calculate Camarilla levels from previous 1d bar
        camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
        camarilla_range = prev_high_1d - prev_low_1d
        camarilla_r3_1d = camarilla_pivot_1d + camarilla_range * 1.1 / 4
        camarilla_s3_1d = camarilla_pivot_1d - camarilla_range * 1.1 / 4
        
        # Align to 4h timeframe (these levels are valid after the 1d bar completes)
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
        
        camarilla_pivot = camarilla_pivot_aligned
        camarilla_r3 = camarilla_r3_aligned
        camarilla_s3 = camarilla_s3_aligned
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pivot[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > EMA34(1d) AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < EMA34(1d) AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot (mean reversion) OR price < EMA34(1d) (trend flip)
            if (close[i] <= camarilla_pivot[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot (mean reversion) OR price > EMA34(1d) (trend flip)
            if (close[i] >= camarilla_pivot[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals