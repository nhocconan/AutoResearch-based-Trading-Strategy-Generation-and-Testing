#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND price > 1w EMA50
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND price < 1w EMA50
# Exit when price crosses middle BB (20-period SMA) OR BB width > 50th percentile (squeeze end)
# Bollinger squeeze indicates low volatility pre-breakout; breakout captures the move
# 1w EMA50 ensures we only trade in direction of higher timeframe trend to avoid whipsaws
# Volume confirmation filters weak breakouts
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_BollingerSqueeze_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2)
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        middle_bb = sma_20
        bb_width = (upper_bb - lower_bb) / middle_bb  # normalized width
    else:
        sma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        middle_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # BB width percentiles for squeeze detection (20th percentile = squeeze, 50th = exit)
    if len(bb_width) >= 50:
        # Use expanding window to avoid look-ahead: percentile of past values
        bb_width_pct = np.full(n, np.nan)
        for i in range(20, n):  # need at least 20 values for BB width
            past_widths = bb_width[20:i+1]  # exclude future
            if len(past_widths) >= 20:
                sorted_widths = np.sort(past_widths[~np.isnan(past_widths)])
                if len(sorted_widths) > 0:
                    idx_20 = max(0, min(len(sorted_widths)-1, int(0.2 * len(sorted_widths))))
                    idx_50 = max(0, min(len(sorted_widths)-1, int(0.5 * len(sorted_widths))))
                    bb_width_pct[i] = sorted_widths[idx_20]  # 20th percentile threshold
                    bb_width_pct[i+len(sorted_widths)//2] = sorted_widths[idx_50]  # 50th percentile (simplified)
        # Simplified: use fixed threshold based on median of first 100 values
        if np.sum(~np.isnan(bb_width[:100])) > 20:
            bb_width_median = np.nanmedian(bb_width[:100])
            bb_width_squeeze = bb_width_median * 0.5  # squeeze when width < 50% of median
            bb_width_exit = bb_width_median * 1.5     # exit when width > 150% of median
        else:
            bb_width_squeeze = 0.1
            bb_width_exit = 0.3
    else:
        bb_width_squeeze = 0.1
        bb_width_exit = 0.3
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > upper BB AND BB squeeze AND price > 1w EMA50 AND volume spike
            if (close[i] > upper_bb[i] and 
                bb_width[i] < bb_width_squeeze and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < lower BB AND BB squeeze AND price < 1w EMA50 AND volume spike
            elif (close[i] < lower_bb[i] and 
                  bb_width[i] < bb_width_squeeze and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < middle BB OR BB width expansion (squeeze end)
            if (close[i] < middle_bb[i] or 
                bb_width[i] > bb_width_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > middle BB OR BB width expansion (squeeze end)
            if (close[i] > middle_bb[i] or 
                bb_width[i] > bb_width_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals