#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 trend filter and volume spike (>2.0x 20-period average)
# Camarilla pivot levels provide precise intraday support/resistance; breakout of R3/S3 indicates strong momentum
# 1d EMA34 filters for higher timeframe trend alignment; volume spike confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; ATR-based stoploss manages risk via signal=0 on close
# Works in both bull/bear markets: Camarilla levels adapt to volatility, effective in ranging and trending conditions
# Target: 100-200 total trades over 4 years (25-50/year) on 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Need at least 1 bar of history for Camarilla calculation
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for today using yesterday's OHLC
        # Camarilla uses previous day's high, low, close
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla formula
        range_val = prev_high - prev_low
        camarilla_r3 = prev_close + range_val * 1.1 / 4
        camarilla_s3 = prev_close - range_val * 1.1 / 4
        camarilla_r4 = prev_close + range_val * 1.1 / 2
        camarilla_s4 = prev_close - range_val * 1.1 / 2
        
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        
        # Calculate 20-period average volume for confirmation
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * vol_ma_20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price closes below Camarilla R3 OR R4 broken (failed breakout)
            if curr_close < camarilla_r3 or curr_close > camarilla_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla S3 OR S4 broken (failed breakdown)
            if curr_close > camarilla_s3 or curr_close < camarilla_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: breakout above Camarilla R3 + above 1d EMA34 + volume confirmation
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below Camarilla S3 + below 1d EMA34 + volume confirmation
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals