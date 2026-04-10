#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla R4 level AND 1d close > 1d EMA50 (uptrend) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla S4 level AND 1d close < 1d EMA50 (downtrend) AND volume > 1.5x 20-period volume SMA
# - Exit: price retreats to Camarilla R3/S3 levels or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Camarilla levels from 1d provide intraday structure, 1d EMA50 filters trend direction, volume confirms breakout strength
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_camarilla_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla levels: 
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    prev_1d_high = df_1d['high'].values
    prev_1d_low = df_1d['low'].values
    prev_1d_close = df_1d['close'].values
    camarilla_r4 = prev_1d_close + ((prev_1d_high - prev_1d_low) * 1.1 / 2)
    camarilla_r3 = prev_1d_close + ((prev_1d_high - prev_1d_low) * 1.1 / 4)
    camarilla_s3 = prev_1d_close - ((prev_1d_high - prev_1d_low) * 1.1 / 4)
    camarilla_s4 = prev_1d_close - ((prev_1d_high - prev_1d_low) * 1.1 / 2)
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Track entry level for exit logic
    entry_level = np.full(n, np.nan)  # Stores which level was broken (R4 or S4)
    
    for i in range(20, n):  # Start after volume SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(volume_sma_20[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_r4 = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakdown_s4 = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        if position == 0:  # Flat - look for entry
            if breakout_r4 and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_level[i] = camarilla_r4_aligned[i]  # Remember we broke R4
            elif breakdown_s4 and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_level[i] = camarilla_s4_aligned[i]  # Remember we broke S4
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price retreats to R3 (take profit) or breaks S4 with volume (stop/reverse)
            exit_to_r3 = close[i] < camarilla_r3_aligned[i]
            reverse_breakdown = breakdown_s4 and vol_confirm
            
            if exit_to_r3 or reverse_breakdown:
                position = 0
                signals[i] = 0.0
                entry_level[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price retreats to S3 (take profit) or breaks R4 with volume (stop/reverse)
            exit_to_s3 = close[i] > camarilla_s3_aligned[i]
            reverse_breakout = breakout_r4 and vol_confirm
            
            if exit_to_s3 or reverse_breakout:
                position = 0
                signals[i] = 0.0
                entry_level[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals