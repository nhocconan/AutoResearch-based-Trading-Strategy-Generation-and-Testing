# 4h_Keltner_Breakout_12hTrend_Volume
# Hypothesis: Keltner Channel breakout on 4h with 12h trend filter and volume confirmation
# Keltner Channels use ATR for dynamic bands, adapting to volatility regimes
# Breakout above upper band in 12h uptrend with volume spike = long
# Breakdown below lower band in 12h downtrend with volume spike = short
# Works in bull/bear markets via trend filter, volatility-adaptive channels reduce false breakouts
# Target: 20-50 trades/year to avoid fee drag
# Uses proven elements: Keltner (volatility channels), trend filter, volume confirmation
# Timeframe: 4h (primary), HTF: 12h (trend)

#!/usr/bin/env python3
name = "4h_Keltner_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate Keltner Channel on 4h data
    # ATR(20) for channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # EMA(20) for midline
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    kc_upper = ema_20 + (2.0 * atr)
    kc_lower = ema_20 - (2.0 * atr)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: 20-period average (~5 days of 4h bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner band with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if close[i] > kc_upper[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner band with volume and 12h downtrend
            elif close[i] < kc_lower[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below midline or volume drops
            if close[i] < ema_20[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above midline or volume drops
            if close[i] > ema_20[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals