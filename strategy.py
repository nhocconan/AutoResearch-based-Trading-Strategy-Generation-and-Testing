# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
# Hypothesis: Use 6h bars with Camarilla pivot levels from daily timeframe.
# Long when price breaks above R3 with volume spike and daily trend up.
# Short when price breaks below S3 with volume spike and daily trend down.
# Uses volume confirmation to avoid false breakouts.
# Daily trend filter avoids counter-trend trades.
# Targets 15-30 trades/year to minimize fee drag.
# Works in bull markets via breakouts and bear markets via trend-following shorts.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla pivot levels from previous day ---
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R3 = typical_price + (range_val * 1.1 / 2)
    S3 = typical_price - (range_val * 1.1 / 2)
    R4 = typical_price + (range_val * 1.1)
    S4 = typical_price - (range_val * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3.values)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4.values)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # --- 1d EMA trend filter ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_1d_values = ema_34_1d.values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_values)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_6h[i]) or 
            np.isnan(S3_6h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA
        price_vs_ema = close[i] > ema_34_1d_aligned[i]
        
        # Breakout signals with volume confirmation
        long_breakout = (high[i] > R3_6h[i]) and vol_spike[i] and price_vs_ema
        short_breakout = (low[i] < S3_6h[i]) and vol_spike[i] and (not price_vs_ema)
        
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S3 or reversal signals
                exit_signal = (low[i] < S3_6h[i]) or (not price_vs_ema)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R3 or reversal signals
                exit_signal = (high[i] > R3_6h[i]) or (price_vs_ema)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals