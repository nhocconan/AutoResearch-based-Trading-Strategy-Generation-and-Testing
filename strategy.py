#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide institutional support/resistance; breakout above R3 or below S3 with
# volume > 2.0x 30-period average and aligned with daily trend (price vs 1d EMA34) captures strong moves.
# Discrete sizing (0.25) targets 75-200 total trades over 4 years to avoid fee drag.
# Works in bull/bear: breakouts capture momentum, volume filter ensures legitimacy, trend filter avoids counter-trend trades.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low),
    #            S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We only need R3 and S3 for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        camarilla_r3 = np.full(len(prices), np.nan)
        camarilla_s3 = np.full(len(prices), np.nan)
    else:
        # Calculate Camarilla levels for each 1d bar
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        camarilla_r3_1d = close_1d + 1.125 * (high_1d - low_1d)
        camarilla_s3_1d = close_1d - 1.125 * (high_1d - low_1d)
        
        # Align to 4h timeframe (completed 1d bar only)
        camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: break above R3 + above 1d EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below S3 + below 1d EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: break below S3 (reversal) or convergence back to mean
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: break above R3 (reversal) or convergence back to mean
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals