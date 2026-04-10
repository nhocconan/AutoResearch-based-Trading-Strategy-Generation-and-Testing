#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d volume spike and choppiness regime filter
# - Long: Williams %R(14) crosses above -80 (oversold reversal) + 1d volume > 1.5x 20-period MA + Chop > 61.8 (ranging market)
# - Short: Williams %R(14) crosses below -20 (overbought reversal) + 1d volume > 1.5x 20-period MA + Chop > 61.8 (ranging market)
# - Exit: Williams %R returns to -50 (mean reversion) or Chop < 38.2 (trending market begins)
# - Position sizing: 0.25 discrete level
# - Works in bull/bear: Williams %R captures reversals in ranging markets, volume confirms participation,
#   Chop filter avoids trending markets where reversals fail. Targets ~40-80 trades/year.

name = "4h_1d_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Williams %R(14) for 4h timeframe
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate Choppiness Index(14) for 4h timeframe
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR and sum of TR over period
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate max high and min low over period
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # Avoid division by zero
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: Chop > 61.8 (ranging market) for reversals
        regime_filter = chop[i] > 61.8
        
        if position == 0:  # Flat - look for Williams %R reversals
            # Long entry: Williams %R crosses above -80 (oversold reversal) + vol confirmation + ranging market
            if i > 0 and williams_r[i-1] <= -80 and williams_r[i] > -80 and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R crosses below -20 (overbought reversal) + vol confirmation + ranging market
            elif i > 0 and williams_r[i-1] >= -20 and williams_r[i] < -20 and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 (mean reversion) OR Chop < 38.2 (trending market begins)
            if position == 1:  # Long position
                if williams_r[i] >= -50 or chop[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50 or chop[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals