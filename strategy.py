#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 levels with 1d EMA34 trend filter and volume spike confirmation
# Camarilla R3/S3 act as strong intraday support/resistance; breaks with volume indicate continuation
# 1d EMA34 filters for intermediate trend alignment (long only above EMA34, short only below)
# Volume spike (2.0x 20-period) confirms momentum validity
# Discrete sizing 0.25 balances profit potential with drawdown control in BTC/ETH bear markets
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    # Camarilla calculation: based on previous day's high, low, close
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    # We use R3/S3 for fade/breakout logic
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    camarilla_r3 = c_12h + ((h_12h - l_12h) * 1.1 / 4)
    camarilla_s3 = c_12h - ((h_12h - l_12h) * 1.1 / 4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for volume MA and 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price closes above R3 with volume
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below S3 with volume
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below EMA34 (trend invalidated) or re-enters Camarilla H3-L3 range
            h3 = camarilla_r3_aligned[i]  # R3 acts as resistance turned support
            l3 = camarilla_s3_aligned[i]  # S3 acts as support turned resistance
            if curr_close < curr_ema_34 or (curr_low > l3 and curr_high < h3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA34 (trend invalidated) or re-enters Camarilla H3-L3 range
            h3 = camarilla_r3_aligned[i]
            l3 = camarilla_s3_aligned[i]
            if curr_close > curr_ema_34 or (curr_low > l3 and curr_high < h3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals