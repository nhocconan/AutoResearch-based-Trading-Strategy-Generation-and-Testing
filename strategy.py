# 4h_Camarilla_Pivot_Breakout_VolumeTrend
# Hypothesis: Price respects Camarilla pivot levels (H4/L4) from the prior 1-day.
# Long when price breaks above H4 with volume > 1.5x average and price above 1-day EMA34 (trend filter).
# Short when price breaks below L4 with volume > 1.5x average and price below 1-day EMA34.
# Exit on opposite breakout or when price returns to the prior 1-day midpoint (mean reversion).
# Uses 4h timeframe with 1d Camarilla levels and EMA34 for trend filter.
# Designed to work in both bull and bear markets by combining breakout logic with mean-reversion exits.
# Target: 20-50 trades per year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1-day close, high, low for Camarilla calculation
    pclose = df_1d['close'].shift(1).values  # Prior day close
    phigh = df_1d['high'].shift(1).values    # Prior day high
    plow = df_1d['low'].shift(1).values      # Prior day low
    
    # Calculate Camarilla levels for prior day
    range_ = phigh - plow
    h4 = pclose + (range_ * 1.1 / 2)
    l4 = pclose - (range_ * 1.1 / 2)
    # Also calculate H3/L3 for potential use
    h3 = pclose + (range_ * 1.1 / 4)
    l3 = pclose - (range_ * 1.1 / 4)
    # Midpoint for mean reversion exit
    pmid = (phigh + plow) / 2
    
    # Calculate 1-day EMA34 for trend filter
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d levels to 4h timeframe
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    pmid_4h = align_htf_to_ltf(prices, df_1d, pmid)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or np.isnan(pmid_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above H4 with volume spike and above EMA34
            if price > h4_4h[i] and vol > 1.5 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 with volume spike and below EMA34
            elif price < l4_4h[i] and vol > 1.5 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint (mean reversion) OR breaks below L4 (invalidates)
            if price < pmid_4h[i] or price < l4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint (mean reversion) OR breaks above H4 (invalidates)
            if price > pmid_4h[i] or price > h4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0