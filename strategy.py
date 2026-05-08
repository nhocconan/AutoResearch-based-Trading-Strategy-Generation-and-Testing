#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 13-period CCI with 12-hour trend filter and volume confirmation
# Long when CCI crosses above -100 (oversold recovery) with 12h EMA(50) uptrend and volume spike
# Short when CCI crosses below +100 (overbought rejection) with 12h EMA(50) downtrend and volume spike
# CCI identifies turning points in both trending and ranging markets. The 12h EMA(50) filter ensures
# we trade with intermediate-term momentum, reducing whipsaw. Volume spike confirms conviction.
# Targets 12-37 trades/year on 6h timeframe to minimize fee drag while capturing meaningful moves.

name = "6h_CCI_OversoldOverbought_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate CCI(14) on 6h data
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=14, min_periods=14).mean()
    tp_mad = pd.Series(typical_price).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - tp_mean.values) / (0.015 * tp_mad.values)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for CCI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(cci[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        cci_val = cci[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: CCI crosses above -100 (from oversold) + 12h uptrend + volume spike
            if i > start_idx:
                prev_cci = cci[i-1]
                if (not np.isnan(prev_cci) and prev_cci <= -100 and cci_val > -100 and
                    close[i] > ema50_12h_val and vol_spike):
                    signals[i] = 0.25
                    position = 1
            # Enter short: CCI crosses below +100 (from overbought) + 12h downtrend + volume spike
            elif i > start_idx:
                prev_cci = cci[i-1]
                if (not np.isnan(prev_cci) and prev_cci >= 100 and cci_val < 100 and
                    close[i] < ema50_12h_val and vol_spike):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: CCI crosses below +100 (overbought) OR 12h trend turns down
            if i > start_idx:
                prev_cci = cci[i-1]
                if (not np.isnan(prev_cci) and prev_cci >= 100 and cci_val < 100) or close[i] < ema50_12h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: CCI crosses above -100 (oversold) OR 12h trend turns up
            if i > start_idx:
                prev_cci = cci[i-1]
                if (not np.isnan(prev_cci) and prev_cci <= -100 and cci_val > -100) or close[i] > ema50_12h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals