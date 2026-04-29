#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA50 trend filter and volume spike confirmation
# Williams %R measures overbought/oversold levels (%R = (Highest High - Close)/(Highest High - Lowest Low) * -100)
# In bull markets (price > 12h EMA50), we look for Williams %R crossing above -50 from oversold with volume confirmation for longs
# In bear markets (price < 12h EMA50), we look for Williams %R crossing below -50 from overbought with volume confirmation for shorts
# Uses strict volume confirmation (>1.8x 24-period average) to reduce false signals
# Designed for ~12-37 trades/year on 6h timeframe to minimize fee drag while capturing momentum reversals
# Works in both bull and bear via 12h EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "6h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 24  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_williams_r = williams_r[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Williams %R crosses below -80 (overbought exhaustion)
            if curr_close < entry_price - 1.5 * curr_atr or curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Williams %R crosses above -20 (oversold exhaustion)
            if curr_close > entry_price + 1.5 * curr_atr or curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 24-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry when price > 12h EMA50 (bullish regime) AND Williams %R crosses above -50 from oversold
            if curr_close > curr_ema50_12h and curr_williams_r > -50 and vol_confirm:
                # Additional confirmation: Williams %R was below -50 in previous bar (crossing up)
                if i > start_idx and williams_r[i-1] <= -50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry when price < 12h EMA50 (bearish regime) AND Williams %R crosses below -50 from overbought
            elif curr_close < curr_ema50_12h and curr_williams_r < -50 and vol_confirm:
                # Additional confirmation: Williams %R was above -50 in previous bar (crossing down)
                if i > start_idx and williams_r[i-1] >= -50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals