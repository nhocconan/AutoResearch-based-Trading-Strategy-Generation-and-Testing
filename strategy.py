#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Exit when price crosses the Camarilla H4/L4 levels (mean reversion zone).
# Uses discrete position sizing (0.25) to balance profit potential and drawdown control.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Works in bull/bear via 1d EMA34 trend filter and strict volume confirmation to avoid false breakouts.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # For 12h timeframe, we use daily OHLC from 1d data
    # Camarilla levels: H4 = C + 1.1*(H-L)*1.1/2, L4 = C - 1.1*(H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C, H, L are from previous day's close, high, low
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(df_1d['high'].values, 1)
    low_1d_shifted = np.roll(df_1d['low'].values, 1)
    # First value will be invalid (rolled), but min_periods in EMA will handle warmup
    
    camarilla_range = high_1d_shifted - low_1d_shifted
    r3_level = close_1d_shifted + camarilla_range * 1.1 / 4.0
    s3_level = close_1d_shifted - camarilla_range * 1.1 / 4.0
    h4_level = close_1d_shifted + camarilla_range * 1.1 * 1.1 / 2.0
    l4_level = close_1d_shifted - camarilla_range * 1.1 * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_level)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_level)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for EMA (34+1 for roll) and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above R3, uptrend (price > 1d EMA34), volume confirmation
            if (curr_high > r3_aligned[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below S3, downtrend (price < 1d EMA34), volume confirmation
            elif (curr_low < s3_aligned[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below H4 (mean reversion zone)
            if curr_close < h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above L4 (mean reversion zone)
            if curr_close > l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals