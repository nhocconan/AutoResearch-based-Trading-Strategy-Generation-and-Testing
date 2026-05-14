# 4h_1d_CamillaBreakout_Volume_Confirm
# Hypothesis: Camarilla pivot levels from 1d (H3/L3) act as strong intraday support/resistance.
# A breakout above H3 or below L3 with volume confirmation (>1.5x average volume) captures
# institutional breakout moves. Works in both bull (breakouts continue) and bear (false breakouts fade quickly,
# but volume helps distinguish real breaks). Uses 4h timeframe to reduce trade frequency.
# Target: 20-40 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    # We need the previous day's OHLC to calculate today's levels
    # Since we're using 4h bars, we'll calculate daily levels and align
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate typical Camarilla levels using previous day's data
    # H3 = C + 1.1*(H-L)/6
    # L3 = C - 1.1*(H-L)/6
    # Where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle NaN
    hl_range = prev_high - prev_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # small value to avoid div by zero
    
    camarilla_h3 = prev_close + 1.1 * hl_range / 6
    camarilla_l3 = prev_close - 1.1 * hl_range / 6
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # 20 for volume avg
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above H3 with volume confirmation
            if price > camarilla_h3_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: breakout below L3 with volume confirmation
            elif price < camarilla_l3_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes back below L3 (mean reversion) or opposite signal
            if price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes back above H3 (mean reversion) or opposite signal
            if price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_CamillaBreakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0