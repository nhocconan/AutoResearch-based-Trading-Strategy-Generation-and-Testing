#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels (R3, S3) act as strong intraday support/resistance
# Breakout above R3 or below S3 with volume confirmation captures momentum
# 1w EMA34 as trend filter: only long when price > EMA34, short when price < EMA34
# Volume spike (2.0x 20-period average) confirms breakout validity
# Discrete sizing 0.25 minimizes fee churn. Works in bull via breakouts with uptrend,
# in bear via breakouts with downtrend. Target: 7-25 trades/year (30-100 total over 4 years).

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L) * 1.1/2, S3 = C - (H-L) * 1.1/2
    typical_price = (high + low + close) / 3.0
    price_range = high - low
    camarilla_pivot = typical_price
    camarilla_r3 = close + price_range * 1.1 / 2.0
    camarilla_s3 = close - price_range * 1.1 / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # warmup for volume MA and 1w EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34 = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND price > 1w EMA34 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND price < 1w EMA34 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below S3 (reversal) OR price drops below EMA34 (trend change)
            if curr_close < camarilla_s3[i] or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 (reversal) OR price rises above EMA34 (trend change)
            if curr_close > camarilla_r3[i] or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals