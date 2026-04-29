#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion + 1d EMA34 trend filter + volume spike
# Williams %R identifies overbought/oversold conditions; 1d EMA34 filters for higher timeframe trend;
# volume confirms reversal strength. Works in bull/bear via trend filter (long only in uptrend, short only in downtrend).
# Target: 20-50 trades/year (80-200 total over 4 years) to avoid fee drag.

name = "4h_WilliamsR_1dEMA34_VolumeSpike_MeanRev_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(50, 34, 20, 14)  # warmup for Williams %R, EMA34, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Fixed stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Williams %R rises above -20 (overbought)
            # 3. Price crosses below 1d EMA34 (trend change)
            if (curr_low <= stop_price or
                curr_williams_r > -20 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Fixed stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * atr_at_entry
            
            # Exit conditions:
            # 1. Stoploss hit
            # 2. Williams %R falls below -80 (oversold)
            # 3. Price crosses above 1d EMA34 (trend change)
            if (curr_high >= stop_price or
                curr_williams_r < -80 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter with volume confirmation to avoid false signals
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: Williams %R below -80 (oversold) + price above 1d EMA34 (uptrend filter)
            if (curr_williams_r < -80 and
                curr_close > curr_ema_34_1d):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: Williams %R above -20 (overbought) + price below 1d EMA34 (downtrend filter)
            elif (curr_williams_r > -20 and
                  curr_close < curr_ema_34_1d):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals