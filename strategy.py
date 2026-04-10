#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Long when Williams %R < -80 (oversold) AND price > 12h EMA20 (uptrend) AND volume > 1.5x average
# - Short when Williams %R > -20 (overbought) AND price < 12h EMA20 (downtrend) AND volume > 1.5x average
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses 12h trend filter to avoid counter-trend trades and volume confirmation to reduce false signals
# - Williams %R is effective in ranging markets (2025+) and catches reversals in bear rallies
# - Tight entry conditions target 20-35 trades/year (80-140 total over 4 years) to minimize fee drag

name = "4h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Pre-compute Williams %R (14-period) on 4h data
    highest_high = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min().values
    close_prices = prices['close'].values
    williams_r = -100 * (highest_high - close_prices) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long: oversold AND 12h uptrend AND volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema20_12h_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: overbought AND 12h downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema20_12h_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion complete)
            if position == 1:  # Long position
                if williams_r[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if williams_r[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals