#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + ATR volatility filter
# - Williams Alligator: Jaw (13-period SMA, 8-bar shift), Teeth (8-period SMA, 5-bar shift), Lips (5-period SMA, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) with volume confirmation and ATR filter
# - Short when Lips < Teeth < Jaw (bearish alignment) with volume confirmation and ATR filter
# - ATR filter: only trade when ATR(14) > 0.3 * ATR(50) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 12h timeframe
# - Williams Alligator works in trending markets (both bull and bear) by identifying trend direction and alignment
# - 1d HTF provides reliable volume confirmation and volatility assessment
# - 12h timeframe balances signal quality and trade frequency to minimize fee drag

name = "12h_1d_williams_alligator_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d volume SMA and ATR
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for ATR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume SMA (20-period)
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Pre-compute 12h Williams Alligator
    # Jaw: 13-period SMA of median price, shifted 8 bars
    # Teeth: 8-period SMA of median price, shifted 5 bars
    # Lips: 5-period SMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    median_series = pd.Series(median_price)
    
    sma_5 = median_series.rolling(window=5, min_periods=5).mean().values
    sma_8 = median_series.rolling(window=8, min_periods=8).mean().values
    sma_13 = median_series.rolling(window=13, min_periods=13).mean().values
    
    # Apply shifts (Alligator specific)
    lips = np.roll(sma_5, 3)   # 5-period SMA shifted 3 bars forward
    teeth = np.roll(sma_8, 5)  # 8-period SMA shifted 5 bars forward
    jaw = np.roll(sma_13, 8)   # 13-period SMA shifted 8 bars forward
    
    # Set NaN for invalid shifted values
    lips[:3] = np.nan
    teeth[:5] = np.nan
    jaw[:8] = np.nan
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average (using 1d aligned volume)
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # ATR filter: trade only when short-term ATR > 0.3 * long-term ATR (avoid low volatility chop)
        atr_filter = atr_14_aligned[i] > 0.3 * atr_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish Alligator alignment + volume confirmation + ATR filter
        if bullish_alignment and vol_confirm and atr_filter:
            enter_long = True
        
        # Short: Bearish Alligator alignment + volume confirmation + ATR filter
        if bearish_alignment and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite Alligator alignment or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bearish alignment OR volatility collapses
            exit_long = bearish_alignment or (not atr_filter)
        elif position == -1:
            # Exit short if bullish alignment OR volatility collapses
            exit_short = bullish_alignment or (not atr_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals