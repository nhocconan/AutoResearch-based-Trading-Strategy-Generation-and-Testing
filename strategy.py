#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# Choppiness index (1d) > 61.8 = ranging market (avoid breakouts), < 38.2 = trending (favor breakouts)
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1dChop_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Choppiness Index (14) ===
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate True Range for 1d
    tr_1d = true_range(high_1d, low_1d, np.roll(close_1d, 1))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first period
    
    # ATR(14) - sum of TR over 14 periods
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max(HH) and Min(LC) over 14 periods
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    range_1d = max_high_1d - min_low_1d
    chop_1d = np.where(
        (range_1d > 0) & (atr_1d > 0),
        100 * np.log10(atr_1d / range_1d) / np.log10(14),
        50  # neutral when undefined
    )
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    # Based on prior day's OHLC
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.25 / 2)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.25 / 2)
    
    # Align 1d indicators to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # === Volume Confirmation: 12h volume > 2.0x 24-period average (24*12h = 12 days) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
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
        curr_chop = chop_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending regimes (CHOP < 38.2) with volume spike
            if curr_chop < 38.2 and curr_volume_spike:
                # Bullish entry: break above R3 with close > R3
                if curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S3 with close < S3
                elif curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (breakout fails) OR chop increases (> 61.8 = ranging)
            if curr_close < curr_r3 or curr_chop > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (breakdown fails) OR chop increases (> 61.8 = ranging)
            if curr_close > curr_s3 or curr_chop > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals