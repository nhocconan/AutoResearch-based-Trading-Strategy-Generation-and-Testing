#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (H3/L3) breakout with 1d EMA50 trend filter and volume spike confirmation
# Camarilla levels provide precise support/resistance based on previous day's range
# Breakout above H3 or below L3 with volume confirmation indicates institutional participation
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear markets by following 1d trend direction via EMA50

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_Volume_v1"
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
    
    # 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h EMA20 for volume filter
    close_s = pd.Series(close)
    vol_series = pd.Series(volume)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # Need 1d EMA50, 12h EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Calculate 12h Camarilla levels from previous completed 12h bar
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla levels
            h3 = prev_close + (range_val * 1.1 / 4)
            l3 = prev_close - (range_val * 1.1 / 4)
            
            if position == 0:  # Flat - look for new entries
                if uptrend:
                    # Long: break above H3 with volume spike
                    if close[i] > h3 and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                elif downtrend:
                    # Short: break below L3 with volume spike
                    if close[i] < l3 and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Avoid sideways markets
            
            elif position == 1:  # Long position
                # Exit: price falls below previous bar's close (loss of momentum)
                if close[i] < prev_close:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit: price rises above previous bar's close (loss of momentum)
                if close[i] > prev_close:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals