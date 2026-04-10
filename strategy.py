#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d close > 1d EMA50 AND volume > 1.5x average
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d close < 1d EMA50 AND volume > 1.5x average
# - Exit when Williams %R crosses back through -50 (mean reversion midpoint) OR volume drops below 0.7x average
# - Uses 1d trend filter to avoid counter-trend trades and volume spike to confirm momentum
# - Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# - Williams %R is effective in ranging markets (2025+) and captures reversals in trending markets

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
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Williams %R(14) on 6h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min().values
    close_prices = prices['close'].values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_prices) / (highest_high - lowest_low),
                          -50)  # neutral when range is zero
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Williams %R crossing above -80 from below (bullish reversal)
            wr_cross_up = (williams_r[i] > -80) and (williams_r[i-1] <= -80)
            # Williams %R crossing below -20 from above (bearish reversal)
            wr_cross_down = (williams_r[i] < -20) and (williams_r[i-1] >= -20)
            
            # Long entry: oversold bounce with volume spike AND 1d uptrend
            if (wr_cross_up and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: overbought rejection with volume spike AND 1d downtrend
            elif (wr_cross_down and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
                
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R crosses back through -50 (mean reversion midpoint)
            # 2. Volume drops below 0.7x average (loss of momentum)
            wr_cross_up_50 = (williams_r[i] > -50) and (williams_r[i-1] <= -50)
            wr_cross_down_50 = (williams_r[i] < -50) and (williams_r[i-1] >= -50)
            
            if position == 1:  # Long position
                if (wr_cross_down_50 or  # Williams %R crossed below -50 (exiting oversold)
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (wr_cross_up_50 or   # Williams %R crossed above -50 (exiting overbought)
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals