#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Bands with width percentile regime filter and volume confirmation.
# Uses Bollinger Bands (20,2) on 1d data to identify volatility regimes.
# In low volatility (BB width < 20th percentile): mean reversion at band touches.
# In high volatility (BB width > 80th percentile): trend following on breakouts.
# Volume confirmation required for all entries. Designed for low trade frequency to avoid fee drag.

name = "6h_BBands_WidthPercentile_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2) on 1d close
    bb_period = 20
    bb_std = 2
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + (bb_std * std_1d)
    lower_bb = sma_1d - (bb_std * std_1d)
    bb_width = upper_bb - lower_bb
    
    # Calculate percentile of BB width using expanding window for regime detection
    # Use 60-day lookback for percentile calculation (approximately 3 months)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=60, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # Define regimes: low volatility < 20th percentile, high volatility > 80th percentile
    # Middle range (20-80) is neutral - no trades
    low_vol_regime = bb_width_pct < 20
    high_vol_regime = bb_width_pct > 80
    
    # Align 1d indicators to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    high_vol_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime)
    
    # Volume confirmation: 6h volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_1d_aligned[i]) or 
            np.isnan(low_vol_aligned[i]) or 
            np.isnan(high_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Low volatility regime: mean reversion at band touches
            if low_vol_aligned[i]:
                # Long when touching lower band
                if low[i] <= lower_bb_aligned[i] and close[i] > lower_bb_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when touching upper band
                elif high[i] >= upper_bb_aligned[i] and close[i] < upper_bb_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # High volatility regime: trend following on breakouts
            elif high_vol_aligned[i]:
                # Long when breaking above upper band with close confirmation
                if close[i] > upper_bb_aligned[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when breaking below lower band with close confirmation
                elif close[i] < lower_bb_aligned[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below SMA (mean reversion) or touches upper band (trend)
            if low_vol_aligned[i]:
                # In low vol: exit at SMA for mean reversion
                if close[i] < sma_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In high vol: exit if price touches upper band (take profit) or closes below lower band (stop)
                if high[i] >= upper_bb_aligned[i]:
                    signals[i] = 0.0  # Take profit at upper band
                    position = 0
                elif close[i] < lower_bb_aligned[i]:
                    signals[i] = 0.0  # Stop loss
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above SMA (mean reversion) or touches lower band (trend)
            if low_vol_aligned[i]:
                # In low vol: exit at SMA for mean reversion
                if close[i] > sma_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In high vol: exit if price touches lower band (take profit) or closes above upper band (stop)
                if low[i] <= lower_bb_aligned[i]:
                    signals[i] = 0.0  # Take profit at lower band
                    position = 0
                elif close[i] > upper_bb_aligned[i]:
                    signals[i] = 0.0  # Stop loss
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals