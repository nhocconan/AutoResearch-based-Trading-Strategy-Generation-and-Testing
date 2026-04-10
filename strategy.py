#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA50 AND volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA50 AND volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 level (mean reversion) OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades in bear markets (2025+)
# - Williams %R is a momentum oscillator that works well in ranging/mean-reverting markets
# - Volume confirmation ensures breakouts have conviction
# - Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# - Focus on BTC/ETH; SOL-only strategies are low value and will be discarded

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 6h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    # Pre-compute aligned 1d data properly
    c_1d = df_1d['close'].values
    
    # Align 1d close to 6h timeframe
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(c_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: oversold AND 1d uptrend AND volume spike
            if (williams_r[i] < -80 and 
                c_1d_aligned[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: overbought AND 1d downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  c_1d_aligned[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for mean reversion exit
            # Exit conditions:
            # 1. Williams %R returns to -50 level (mean reversion complete)
            # 2. Volume drops below 0.7x average (loss of momentum)
            if position == 1:  # Long position
                if (williams_r[i] > -50 or vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] < -50 or vol_weak[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals