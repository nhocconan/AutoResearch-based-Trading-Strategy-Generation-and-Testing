#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter + volume confirmation
# Camarilla levels provide high-probability reversal points; breakouts beyond R3/S3 indicate strong momentum
# Weekly EMA50 ensures trades align with higher timeframe trend (avoid counter-trend in strong moves)
# Volume confirmation validates breakout strength
# Works in bull/bear/range by trading breakouts only when aligned with weekly trend
# Target: 10-20 trades/year (40-80 total over 4 years)

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Need prior day's OHLC for Camarilla calculation
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous day
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        # Avoid division by zero in case phigh == plow
        if phigh == plow:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Calculate Camarilla levels
        range_val = phigh - plow
        r3 = pclose + (range_val * 1.1 / 4)
        s3 = pclose - (range_val * 1.1 / 4)
        r4 = pclose + (range_val * 1.1 / 2)
        s4 = pclose - (range_val * 1.1 / 2)
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema50_1w_aligned[i]
        
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = curr_volume > (1.8 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above R3 with volume confirmation
            if curr_close > r3 and volume_confirm and curr_close > curr_ema50_1w:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below S3 with volume confirmation
            elif curr_close < s3 and volume_confirm and curr_close < curr_ema50_1w:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:  # Long position - exit conditions
            # Exit when: price closes below R3 (breakout failed) OR crosses below weekly EMA50
            if curr_close < r3 or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position - exit conditions
            # Exit when: price closes above S3 (breakout failed) OR crosses above weekly EMA50
            if curr_close > s3 or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals