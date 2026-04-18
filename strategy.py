# NOTE: This strategy is being re-run with the same parameters as experiment #60278, which resulted in a loss of -0.079.  
# Despite the previous loss, we are re-running it to confirm consistency or rule out flukes.  
# Any insights gained will inform future strategy development.  

#!/usr/bin/env python3
"""
1d_RVOL_MeanReversion_v1
Hypothesis: Trade reversals on extreme volume spikes combined with RSI mean reversion on daily timeframe.
Enter long when daily RSI < 30 and volume > 3x 20-day average volume (oversold with panic selling).
Enter short when daily RSI > 70 and volume > 3x 20-day average volume (overbought with buying frenzy).
Exit when RSI returns to neutral range (40-60) or volume normalizes.
Uses 1-week trend filter: only take longs when price > weekly EMA20, shorts when price < weekly EMA20.
Designed to work in both bull (buy dips) and bear (sell rallies) markets by fading extreme moves.
Targets 10-25 trades/year via rare confluence of extreme RSI + extreme volume + trend filter.
"""

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
    
    # Get daily data for RSI and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Daily RSI(14)
    rsi_period = 14
    close_1d = df_1d['close'].values
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Daily 20-day average volume
    vol_ma_period = 20
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    
    if len(vol_1d) >= vol_ma_period:
        for i in range(vol_ma_period, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i - vol_ma_period:i])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(20)
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (20 + 1)) + (ema_1w[i-1] * (19 / (20 + 1)))
    
    # Align daily indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # But we still use the alignment function for consistency with HTF->LTF conversion
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align weekly EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period, vol_ma_period) + 5  # Ensure indicators are warm
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current daily volume > 3x 20-day average
        vol_spike = volume[i] > 3.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volume spike + price > weekly EMA (uptrend filter)
            if rsi_1d_aligned[i] < 30 and vol_spike and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + volume spike + price < weekly EMA (downtrend filter)
            elif rsi_1d_aligned[i] > 70 and vol_spike and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (>40) OR volume normalizes OR price breaks below weekly EMA
            if (rsi_1d_aligned[i] > 40 or 
                volume[i] <= 1.5 * vol_ma_1d_aligned[i] or 
                close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (<60) OR volume normalizes OR price breaks above weekly EMA
            if (rsi_1d_aligned[i] < 60 or 
                volume[i] <= 1.5 * vol_ma_1d_aligned[i] or 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RVOL_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0