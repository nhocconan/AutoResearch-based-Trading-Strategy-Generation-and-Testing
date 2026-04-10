#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) on 6h: oversold < -80 for long, overbought > -20 for short
# - 1d EMA(50) trend filter: only long when 1d close > EMA50, only short when 1d close < EMA50
# - Volume confirmation: require volume > 1.5x 20-period average to avoid low-conviction trades
# - Exit when Williams %R reverts to mean (-50 level) or trend changes
# - Targets 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# - Works in both bull/bear markets by aligning with 1d trend and fading extremes

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
    
    # Pre-compute Williams %R on 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
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
            # Long when oversold AND 1d uptrend AND volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when overbought AND 1d downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for mean reversion exit
            # Exit when Williams %R reverts to mean (-50) or trend changes
            if position == 1:  # Long position
                if (williams_r[i] >= -50 or 
                    prices['close'].iloc[i] < ema50_1d_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] <= -50 or 
                    prices['close'].iloc[i] > ema50_1d_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals