#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and choppiness regime filter
# - Long when Williams %R(14) crosses above -80 (oversold) + 1d volume > 1.8x 20-period volume SMA + CHOP > 61.8 (range market)
# - Short when Williams %R(14) crosses below -20 (overbought) + 1d volume > 1.8x 20-period volume SMA + CHOP > 61.8 (range market)
# - Exit: Williams %R returns to opposite extreme (-20 for long, -80 for short) or reverse signal
# - Position sizing: 0.25 discrete level
# - Williams %R identifies exhaustion points in ranging markets, volume confirms participation, CHOP filter ensures proper market structure
# - Works in bull/bear: mean reversion effective in both regimes when volatility is contained
# - 4h timeframe targets 20-50 trades/year with strict entry conditions

name = "4h_1d_williamsr_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max().values
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    # True Range
    tr1 = np.maximum(df_1d['high'] - df_1d['low'], 
                     np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr1[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    # Choppiness Index: 100 * log10(sum(tr1)/(hh-ll)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero and invalid values
    chop_raw = np.where((hh - ll) == 0, 50, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)
    chop = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period SMA (spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Williams %R conditions
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        # Long when WR crosses above -80 from below (exiting oversold)
        long_signal = (wr_prev <= -80) and (wr > -80)
        # Short when WR crosses below -20 from above (exiting overbought)
        short_signal = (wr_prev >= -20) and (wr < -20)
        
        # Exit conditions: WR returns to opposite extreme
        long_exit = wr >= -20  # WR reaches overbought territory
        short_exit = wr <= -80  # WR reaches oversold territory
        
        # Entry conditions
        long_entry = long_signal and vol_confirm and ranging_market
        short_entry = short_signal and vol_confirm and ranging_market
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals