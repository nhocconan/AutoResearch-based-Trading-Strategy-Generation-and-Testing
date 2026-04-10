#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Camarilla pivot breakout with volume confirmation and ATR filter
# - Weekly Camarilla levels (R3, S3, PP) from prior completed weekly bar
# - Long when price closes above weekly R3 with volume > 1.5x 20-bar average AND ATR(14) < ATR(50)
# - Short when price closes below weekly S3 with volume > 1.5x 20-bar average AND ATR(14) < ATR(50)
# - Exit when price returns to weekly pivot point (PP)
# - ATR condition ensures we trade only in low volatility regimes to avoid whipsaws
# - Weekly timeframe provides clean structure; daily timeframe allows sufficient trades
# - Volume confirmation filters false breakouts
# - Targets ~15 trades/year (60 total over 4 years) to minimize fee drag

name = "1d_weekly_camarilla_breakout_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly indicators for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w = typical_price_1w  # PP = (H+L+C)/3
    range_1w = high_1w - low_1w
    r3_1w = pp_1w + (range_1w * 1.1 / 2.0)
    s3_1w = pp_1w - (range_1w * 1.1 / 2.0)
    
    # Align weekly Camarilla levels to 1d timeframe (completed weekly bar only)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute ATR filter: ATR(14) < ATR(50) for low volatility regime
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR calculations
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_volatility = atr_14 < atr_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(volume_20_avg[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price closes above weekly R3 with volume spike and low volatility
            if (prices['close'].iloc[i] > r3_1w_aligned[i] and 
                vol_spike.iloc[i] and 
                low_volatility[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price closes below weekly S3 with volume spike and low volatility
            elif (prices['close'].iloc[i] < s3_1w_aligned[i] and 
                  vol_spike.iloc[i] and 
                  low_volatility[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to weekly pivot point (PP)
            if position == 1 and prices['close'].iloc[i] < pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals