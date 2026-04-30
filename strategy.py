#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla H3/L3 levels. Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe. Uses 1d HTF for trend filter to avoid look-ahead.
# Works in bull/bear via 1d EMA34 trend filter and volume confirmation - avoids SOL-only bias.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar (H1, L1, C1)
    # We need the previous completed 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: H3, L3, H4, L4, H5, L5, H6, L6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # H3 = C + (H-L)*1.1/2, L3 = C - (H-L)*1.1/2
    # Actually Camarilla: R4 = C+(H-L)*1.1/2, R3 = C+(H-L)*1.1/4, S3 = C-(H-L)*1.1/4, S4 = C-(H-L)*1.1/2
    # We'll use R3 and S3 for entry, H3 and L3 for exit (as per winning patterns)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + rng * 1.1 / 4
    camarilla_s3 = close_1d_vals - rng * 1.1 / 4
    camarilla_h3 = close_1d_vals + rng * 1.1 / 2
    camarilla_l3 = close_1d_vals - rng * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R3, uptrend (price > 1d EMA34), volume confirmation
            if (curr_high > camarilla_r3_aligned[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Camarilla S3, downtrend (price < 1d EMA34), volume confirmation
            elif (curr_low < camarilla_s3_aligned[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Camarilla H3 (take profit) or reverses below S3 (stop)
            if curr_close < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Camarilla L3 (take profit) or reverses above R3 (stop)
            if curr_close > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals