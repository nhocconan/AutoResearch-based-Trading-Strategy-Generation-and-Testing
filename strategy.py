#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band reversal with 12h Bollinger width regime filter and volume confirmation.
# Long when: Price touches lower BB(20,2) AND 12h BW < 30th percentile (low volatility regime) AND volume > 1.3x 20-period average
# Short when: Price touches upper BB(20,2) AND 12h BW < 30th percentile AND volume > 1.3x 20-period average
# Exit when: Price crosses back to middle BB (20-period SMA)
# Works in ranging markets by capturing mean reversion during low volatility periods.
# Avoids choppy markets by requiring low volatility regime (tight Bollinger Bands on higher timeframe).
# Target: 20-35 trades/year per symbol.
name = "6h_BB_Reversal_12hBW_Regime_Volume"
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
    
    # Get 12h Bollinger Width (BBW) regime ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma_20_12h + 2 * std_20_12h
    lower_bb_12h = sma_20_12h - 2 * std_20_12h
    bbw_12h = (upper_bb_12h - lower_bb_12h) / sma_20_12h  # Bollinger Width as % of SMA
    # Calculate 30th percentile of BBW over lookback period
    bbw_pct_30 = pd.Series(bbw_12h).rolling(window=100, min_periods=100).quantile(0.30).values
    bbw_12h_aligned = align_htf_to_ltf(prices, df_12h, bbw_12h)
    bbw_pct_30_aligned = align_htf_to_ltf(prices, df_12h, bbw_pct_30)
    
    # 6h Bollinger Bands for entry/exit
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20  # 20-period SMA
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 100)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bbw_12h_aligned[i]) or np.isnan(bbw_pct_30_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_regime = bbw_12h_aligned[i] < bbw_pct_30_aligned[i]  # Low volatility regime
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price at or below lower BB + low vol regime + volume spike
            if price <= lower_bb[i] and bb_width_regime and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Price at or above upper BB + low vol regime + volume spike
            elif price >= upper_bb[i] and bb_width_regime and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back to middle BB
            if price >= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back to middle BB
            if price <= middle_bb[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals