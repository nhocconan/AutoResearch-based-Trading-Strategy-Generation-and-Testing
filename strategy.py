#!/usr/bin/env python3
# 4h_cci_breakout_1d_trend_volume_v2
# Hypothesis: CCI breakout with daily trend and volume confirmation on 4h timeframe.
# Long when CCI crosses above +100 and price > daily EMA200 with volume > 1.5x average.
# Short when CCI crosses below -100 and price < daily EMA200 with volume > 1.5x average.
# Exit on opposite CCI cross or when volume drops below average.
# Designed to capture strong trends with volume confirmation to reduce whipsaw.
# Tightened entry conditions to reduce trade frequency and avoid overtrading.
# Target: 75-150 total trades over 4 years (~19-38/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # CCI calculation (20-period)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    tp_mean = typical_price.rolling(window=20, min_periods=20).mean()
    tp_mad = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    cci = cci.values
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(cci[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 or volume drops below average
            if cci[i] < 100 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 or volume drops below average
            if cci[i] > -100 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Trend filter: price vs daily EMA200
            price_above_ema = close[i] > ema_200_1d_aligned[i]
            price_below_ema = close[i] < ema_200_1d_aligned[i]
            
            # CCI breakout entries - stricter: require previous close also confirms trend
            if cci[i] > 100 and price_above_ema and volume_ok:
                # Additional confirmation: previous CCI was below +100 to confirm breakout
                # AND previous close also above EMA to avoid whipsaw
                if i > 0 and cci[i-1] <= 100 and close[i-1] > ema_200_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif cci[i] < -100 and price_below_ema and volume_ok:
                # Additional confirmation: previous CCI was above -100 to confirm breakdown
                # AND previous close also below EMA to avoid whipsaw
                if i > 0 and cci[i-1] >= -100 and close[i-1] < ema_200_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals