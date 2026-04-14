#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R mean reversion with Bollinger Band squeeze filter.
# Long when Williams %R < -80 (oversold) AND price < Bollinger lower band AND Bollinger width < 20th percentile (squeeze).
# Short when Williams %R > -20 (overbought) AND price > Bollinger upper band AND Bollinger width < 20th percentile (squeeze).
# Exit when Williams %R crosses -50 (mean reversion complete) OR Bollinger width expands above 50th percentile.
# Williams %R identifies extremes, Bollinger squeeze filters for low volatility breakouts, mean reversion captures snapbacks.
# Target: 30-60 trades/year per symbol (120-240 total over 4 years) to balance opportunity with fee control.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load daily data ONCE for Williams %R and Bollinger Bands
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Williams %R (14-period)
    lookback_wr = 14
    highest_high = np.full_like(high_daily, np.nan)
    lowest_low = np.full_like(low_daily, np.nan)
    
    for i in range(lookback_wr - 1, len(high_daily)):
        highest_high[i] = np.max(high_daily[i - lookback_wr + 1:i + 1])
        lowest_low[i] = np.min(low_daily[i - lookback_wr + 1:i + 1])
    
    williams_r = np.full_like(close_daily, np.nan)
    for i in range(lookback_wr - 1, len(close_daily)):
        if highest_high[i] - lowest_low[i] != 0:
            williams_r[i] = (highest_high[i] - close_daily[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate Bollinger Bands (20-period, 2 std)
    lookback_bb = 20
    std_dev = 2
    sma = np.full_like(close_daily, np.nan)
    bb_upper = np.full_like(close_daily, np.nan)
    bb_lower = np.full_like(close_daily, np.nan)
    bb_width = np.full_like(close_daily, np.nan)
    
    for i in range(lookback_bb - 1, len(close_daily)):
        sma[i] = np.mean(close_daily[i - lookback_bb + 1:i + 1])
        std = np.std(close_daily[i - lookback_bb + 1:i + 1])
        bb_upper[i] = sma[i] + std * std_dev
        bb_lower[i] = sma[i] - std * std_dev
        if sma[i] != 0:
            bb_width[i] = (bb_upper[i] - bb_lower[i]) / sma[i] * 100  # percentage width
        else:
            bb_width[i] = 0
    
    # Calculate Bollinger width percentiles for squeeze filter
    # We'll calculate 20th and 50th percentiles dynamically
    bb_width_valid = bb_width[~np.isnan(bb_width)]
    if len(bb_width_valid) >= 20:
        bb_width_20th = np.percentile(bb_width_valid, 20)
        bb_width_50th = np.percentile(bb_width_valid, 50)
    else:
        bb_width_20th = 5.0  # default squeeze threshold
        bb_width_50th = 10.0  # default expansion threshold
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    bb_upper_aligned = align_htf_to_ltf(prices, df_daily, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_daily, bb_lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_daily, bb_width)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(lookback_wr, lookback_bb)  # Need both indicators
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion entries during Bollinger squeeze
            # Long: oversold (Williams %R < -80) AND price below lower band AND squeeze (width < 20th percentile)
            if (williams_r_aligned[i] < -80 and 
                close[i] < bb_lower_aligned[i] and
                bb_width_aligned[i] < bb_width_20th):
                position = 1
                signals[i] = position_size
            # Short: overbought (Williams %R > -20) AND price above upper band AND squeeze (width < 20th percentile)
            elif (williams_r_aligned[i] > -20 and 
                  close[i] > bb_upper_aligned[i] and
                  bb_width_aligned[i] < bb_width_20th):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: mean reversion complete OR volatility expansion
            if (williams_r_aligned[i] > -50 or  # Williams %R crossed midpoint
                bb_width_aligned[i] > bb_width_50th):  # volatility expanded
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: mean reversion complete OR volatility expansion
            if (williams_r_aligned[i] < -50 or  # Williams %R crossed midpoint
                bb_width_aligned[i] > bb_width_50th):  # volatility expanded
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_BollingerSqueeze_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0