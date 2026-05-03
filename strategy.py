#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d HMA50 trend filter and volume spike confirmation.
# Uses 4h timeframe for optimal trade frequency (target 20-50 trades/year/symbol), 1d for HTF direction and pivot calculation.
# Breakouts above R4 (long) or below S4 (short) with volume confirmation and trend alignment.
# ATR-based trailing stop for risk management. Discrete sizing 0.30 to balance return and drawdown.
# Target: 80-200 total trades over 4 years (20-50/year) to minimize fee drag while capturing Camarilla edge.

name = "4h_Camarilla_R4_S4_1dHMA50_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation, trend filter, and volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior completed 1d bar's OHLC for Camarilla calculation
    prior_close = np.roll(df_1d['close'].values, 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Calculate Camarilla levels for prior 1d bar
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_r4 = prior_close + (prior_high - prior_low) * 1.1 / 2
    camarilla_s4 = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d HMA50 trend filter
    close_1d = df_1d['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 50 // 2
    sqrt_len = int(np.sqrt(50))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.full_like(close_1d, np.nan)
    wma_full = np.full_like(close_1d, np.nan)
    for i in range(half_len, len(close_1d)):
        wma_half[i] = wma(close_1d[i-half_len+1:i+1], half_len)
    for i in range(50, len(close_1d)):
        wma_full[i] = wma(close_1d[i-50+1:i+1], 50)
    
    hma_50 = 2 * wma_half - wma_full
    hma_50 = np.concatenate([np.full(half_len-1, np.nan), hma_50[half_len-1:]])
    hma_50_aligned = align_htf_to_ltf(prices, df_1d, hma_50)
    
    # Calculate 1d volume regime (high volume when current volume > 1.5x 20-period MA)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_regime = vol_1d > (1.5 * vol_ma_1d)  # High volume regime
    
    # Align volume regime to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate ATR(14) for 4h data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):
        # Get current values
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema_trend = hma_50_aligned[i]
        vol_reg = vol_regime_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(r4) or np.isnan(s4) or np.isnan(ema_trend) or np.isnan(vol_reg) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 4h volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Entry conditions
        # Long: break above R4 with volume spike, above 1d HMA50, and in high volume regime
        long_entry = (close[i] > r4) and volume_spike and (close[i] > ema_trend) and vol_reg
        # Short: break below S4 with volume spike, below 1d HMA50, and in high volume regime
        short_entry = (close[i] < s4) and volume_spike and (close[i] < ema_trend) and vol_reg
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.30
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.30
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals