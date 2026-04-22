#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h RSI(2) mean reversion with 1d EMA200 trend filter and volume confirmation
    # RSI(2) captures extreme short-term reversals (<10 oversold, >90 overbought)
    # 1d EMA200 ensures trading with the higher timeframe trend
    # Volume spike confirms institutional participation
    # This combination works in both bull (buy dips) and bear (sell rallies) markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA200 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) calculation
    def rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(series)
        avg_loss = np.zeros_like(series)
        
        # Initialize first average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    
    rsi2 = rsi(close, 2)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi2[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) oversold (<10) + volume spike + price above 1d EMA200
            if rsi2[i] < 10 and vol_spike[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) overbought (>90) + volume spike + price below 1d EMA200
            elif rsi2[i] > 90 and vol_spike[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI(2) returns to neutral (50) or trend reversal vs 1d EMA200
            if position == 1:
                if rsi2[i] > 50 or close[i] < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi2[i] < 50 or close[i] > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_RSI2_MeanReversion_1dEMA200_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0