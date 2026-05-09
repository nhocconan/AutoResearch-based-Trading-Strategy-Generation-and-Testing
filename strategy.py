#!/usr/bin/env python3
# 4h_VWAP_Reversion_with_12hTrend_Filter
# Hypothesis: Combines 4h VWAP mean reversion with 12h trend filter and volume confirmation.
# Long when price < VWAP (oversold) and 12h EMA50 is rising; short when price > VWAP (overbought) and 12h EMA50 is falling.
# Uses volume spike for confirmation. Designed to work in both trending and ranging markets.
# Target: 20-35 trades/year per symbol with disciplined risk.

name = "4h_VWAP_Reversion_with_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50[i] = alpha * close_12h[i] + (1 - alpha) * ema_50[i-1]
    
    # Calculate EMA slope for trend direction
    ema_slope = np.full_like(ema_50, np.nan)
    if len(ema_50) >= 2:
        ema_slope[1:] = ema_50[1:] - ema_50[:-1]
    
    # Align 12h indicators to 4h timeframe
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Calculate 4h VWAP (typical price * volume) / volume
    typical_price = (high + low + close) / 3
    vwap_num = np.full_like(typical_price, np.nan)
    vwap_den = np.full_like(typical_price, np.nan)
    
    if len(typical_price) >= 1:
        vwap_num[0] = typical_price[0] * volume[0]
        vwap_den[0] = volume[0]
        for i in range(1, len(typical_price)):
            vwap_num[i] = vwap_num[i-1] + typical_price[i] * volume[i]
            vwap_den[i] = vwap_den[i-1] + volume[i]
    
    vwap = np.full_like(typical_price, np.nan)
    valid_den = vwap_den != 0
    vwap[valid_den] = vwap_num[valid_den] / vwap_den[valid_den]
    
    # Calculate VWAP deviation (price - VWAP)
    vwap_deviation = typical_price - vwap
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_slope_aligned[i]) or np.isnan(vwap_deviation[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2x average
        volume_confirm = volume_ratio[i] > 2.0
        
        if position == 0:
            # Enter long: Price below VWAP (oversold) AND 12h trend up AND volume confirmation
            if vwap_deviation[i] < 0 and ema_slope_aligned[i] > 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: Price above VWAP (overbought) AND 12h trend down AND volume confirmation
            elif vwap_deviation[i] > 0 and ema_slope_aligned[i] < 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: Price crosses above VWAP OR trend turns down
            if vwap_deviation[i] > 0 or ema_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: Price crosses below VWAP OR trend turns up
            if vwap_deviation[i] < 0 or ema_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals