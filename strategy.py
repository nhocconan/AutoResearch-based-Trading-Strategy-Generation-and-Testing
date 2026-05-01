#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1w EMA50 trend filter and volume confirmation (>2.0x 20-bar MA)
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend direction and entry timing.
# 1w EMA50 provides strong multi-timeframe trend alignment to reduce whipsaws in ranging markets.
# Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) with strong BTC/ETH performance in both bull and bear markets.

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) on 1w close
    ema_1w_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shifts
    # Jaw: 13-period SMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (need 13+8=21 for Jaw, plus shifts)
    start_idx = 21
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish trend
            # Lips < Teeth < Jaw = bearish trend
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish alignment, price above 1w EMA, and volume confirmation
            if bullish_alignment and curr_close > ema_1w_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below 1w EMA, and volume confirmation
            elif bearish_alignment and curr_close < ema_1w_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish Alligator alignment or price below 1w EMA
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            if bearish_alignment or curr_close < ema_1w_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish Alligator alignment or price above 1w EMA
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if bullish_alignment or curr_close > ema_1w_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals