#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d ATR regime filter + volume confirmation
# - Long when price breaks above Camarilla H3 (1d) AND 1d ATR(14) > 20-period MA AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 (1d) AND 1d ATR(14) > 20-period MA AND volume > 1.5x 20-period average
# - Exit when price crosses Camarilla Pivot Point (PP) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla pivots from 1d provide institutional support/resistance levels
# - ATR regime filter ensures we only trade in sufficient volatility environments (avoids chop)
# - Volume confirmation reduces false breakouts

name = "4h_1d_camarilla_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas:
    # PP = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    # Calculate daily Camarilla levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_l3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_l4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Pre-compute 1d ATR(14) for regime filter
    # ATR = True Range smoothed (Wilder's smoothing)
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    # Wilder's smoothing: ATR today = (ATR yesterday * 13 + TR today) / 14
    atr_1d = np.full_like(true_range, np.nan)
    atr_1d[13] = np.nanmean(true_range[1:14])  # Seed with simple average of first 14
    for i in range(14, len(true_range)):
        if not np.isnan(atr_1d[i-1]):
            atr_1d[i] = (atr_1d[i-1] * 13 + true_range[i]) / 14
    
    # ATR regime filter: ATR > 20-period MA of ATR
    atr_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_regime = atr_1d > atr_ma
    
    # Align HTF indicators to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla R3 AND ATR regime AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                atr_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND ATR regime AND volume spike
            elif (close[i] < camarilla_l3_aligned[i] and 
                  atr_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla Pivot Point OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_pp_aligned[i] or close[i] < camarilla_l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_pp_aligned[i] or close[i] > camarilla_r3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals