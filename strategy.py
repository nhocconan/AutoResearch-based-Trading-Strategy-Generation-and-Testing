# 4h_bollinger_breakout_volume_v1
# Strategy: Bollinger Band breakout with volume confirmation and Bollinger Width regime filter.
# Long when price breaks above upper BB with volume > 1.5x avg and BW < 50th percentile (low volatility).
# Short when price breaks below lower BB with volume > 1.5x avg and BW < 50th percentile.
# Exit when price crosses back inside BB or volatility expands (BW > 70th percentile).
# Designed to capture volatility breakouts in both trending and ranging markets.
# Target: 20-50 trades/year per symbol.

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_bollinger_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_ma = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + bb_std * bb_std_dev
    bb_lower = bb_ma - bb_std * bb_std_dev
    
    # Bollinger Width (BW) for volatility regime
    bb_width = (bb_upper - bb_lower) / bb_ma
    # Percentile rank of BB width over last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back inside BB or volatility expands (BW > 70th percentile)
            if close[i] < bb_ma[i] or bb_width_percentile[i] > 0.7:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses back inside BB or volatility expands (BW > 70th percentile)
            if close[i] > bb_ma[i] or bb_width_percentile[i] > 0.7:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Low volatility regime: BB width below 50th percentile
            low_vol = bb_width_percentile[i] < 0.5
            
            # Long entry: price breaks above upper BB with volume and low volatility
            if close[i] > bb_upper[i] and vol_confirm[i] and low_vol:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower BB with volume and low volatility
            elif close[i] < bb_lower[i] and vol_confirm[i] and low_vol:
                position = -1
                signals[i] = -0.25
    
    return signals