#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h for signal direction (EMA50 trend) and 1h for precise entry timing
# Breakout above R1 or below S1 with volume spike indicates momentum
# 4h EMA50 > EMA200 ensures alignment with medium-term trend to avoid whipsaw
# Volume spike (2.0x 24-period average) confirms participation
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Calculate 4h EMA50 and EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # EMA50 and EMA200 on 4h
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
    ema50_gt_ema200 = ema_50 > ema_200
    ema50_lt_ema200 = ema_50 < ema_200
    
    # Align trend filters to 1h timeframe
    ema50_gt_ema200_aligned = align_htf_to_ltf(prices, df_4h, ema50_gt_ema200.astype(float))
    ema50_lt_ema200_aligned = align_htf_to_ltf(prices, df_4h, ema50_lt_ema200.astype(float))
    
    # Calculate 1d Camarilla pivot levels (R1, S1) - using prior day's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels based on prior day's OHLC
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.125 / 4)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.125 / 4)
    
    # Align to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*1h = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 24)  # warmup for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_gt_ema200_aligned[i]) or np.isnan(ema50_lt_ema200_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_ema50_gt_ema200 = bool(ema50_gt_ema200_aligned[i])
        curr_ema50_lt_ema200 = bool(ema50_lt_ema200_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above R1 with 4h uptrend
                if curr_close > curr_r1 and curr_ema50_gt_ema200:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S1 with 4h downtrend
                elif curr_close < curr_s1 and curr_ema50_lt_ema200:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below S1 (mean reversion) OR trend weakens
            if curr_close < curr_s1 or not curr_ema50_gt_ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 (mean reversion) OR trend weakens
            if curr_close > curr_r1 or not curr_ema50_lt_ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals