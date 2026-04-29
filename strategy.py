#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter + volume confirmation + ATR-based trailing stop
# Camarilla levels provide institutional support/resistance; weekly EMA50 ensures alignment with higher timeframe trend;
# volume spike confirms institutional participation; ATR trailing stop adapts to volatility in both bull/bear markets.
# Target: 20-30 trades/year (80-120 total over 4 years) to balance opportunity with fee drag minimization.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_ATRTrail_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss and volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day (R3/S3)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    camarilla_r3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.0 * (prev_high - prev_low)
    
    # Volume confirmation: volume > 1.8x 20-period average (tighter for fewer, higher quality trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    max_high_since_entry = 0.0
    min_low_since_entry = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, volume, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_open = open_price[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry
            max_high_since_entry = max(max_high_since_entry, curr_high)
            # ATR-based trailing stop (2.5x ATR)
            trail_stop = max_high_since_entry - 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Trailing stoploss hit
            # 2. Price closes below 1w EMA50 (trend change)
            # 3. Price drops below Camarilla S3 (breakout failed)
            if (curr_low <= trail_stop or
                curr_close < curr_ema_50_1w or
                curr_close < curr_s3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry
            min_low_since_entry = min(min_low_since_entry, curr_low)
            # ATR-based trailing stop (2.5x ATR)
            trail_stop = min_low_since_entry + 2.5 * curr_atr
            
            # Exit conditions:
            # 1. Trailing stoploss hit
            # 2. Price closes above 1w EMA50 (trend change)
            # 3. Price rises above Camarilla R3 (breakout failed)
            if (curr_high >= trail_stop or
                curr_close > curr_ema_50_1w or
                curr_close > curr_r3):
                signals[i] = 0.0
                position = 0
                max_high_since_entry = 0.0
                min_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 + above 1w EMA50 + volume confirm
            if (curr_close > curr_r3 and
                curr_close > curr_ema_50_1w and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            # Short entry: price breaks below Camarilla S3 + below 1w EMA50 + volume confirm
            elif (curr_close < curr_s3 and
                  curr_close < curr_ema_50_1w and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                max_high_since_entry = curr_high
                min_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals