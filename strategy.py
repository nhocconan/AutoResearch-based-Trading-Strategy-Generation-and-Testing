#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w EMA50 trend filter with volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1w EMA50 AND volume > 1.5x 20-bar avg.
# Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 1w EMA50 AND volume > 1.5x 20-bar avg.
# Exit when momentum weakens (Bull Power <= 0 for long, Bear Power <= 0 for short).
# Uses 1w EMA50 for higher timeframe trend alignment, targeting 12-37 trades/year on 6h.
# Works in bull markets via long entries with trend alignment and in bear markets via short entries with trend alignment.

name = "6h_ElderRay_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray components: need EMA13 of high/low/close
    # Use 13-period EMA for the Elder Ray calculation
    ema_13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13_close  # Bull Power = High - EMA13
    bear_power = ema_13_close - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA13 and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1w EMA50 AND volume spike
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                close[i] > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 1w EMA50 AND volume spike
            elif (curr_bear_power > 0 and 
                  curr_bull_power < 0 and 
                  close[i] < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: bullish momentum weakens (Bull Power <= 0)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: bearish momentum weakens (Bear Power <= 0)
            if curr_bear_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals