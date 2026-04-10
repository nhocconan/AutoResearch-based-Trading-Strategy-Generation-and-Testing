#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Volume Spike Regime + Price Channel Filter
# - Long when Alligator Jaw < Teeth < Lips (bullish alignment) AND price > Donchian(20) upper band AND 1d volume > 2.0x 20-period median volume
# - Short when Alligator Jaw > Teeth > Lips (bearish alignment) AND price < Donchian(20) lower band AND 1d volume > 2.0x 20-period median volume
# - Exit when Alligator lines cross (Jaw-Teeth or Teeth-Lips crossover) indicating trend weakening
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Williams Alligator identifies trend alignment with smoothed moving averages
# - Volume spike regime ensures we trade during institutional participation
# - Donchian channel provides objective breakout levels
# - Works in both bull (trend following) and bear (mean reversion during alignment breaks) markets

name = "4h_1d_alligator_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator alignment conditions
    bullish_alignment = (jaw < teeth) & (teeth < lips)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Pre-compute 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation (20-period median)
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Pre-compute 1d volume regime (HTF)
    vol_1d = df_1d['volume'].values
    vol_median_1d_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    high_vol_regime_1d = vol_1d > (1.5 * vol_median_1d_20)  # High volume days on daily
    
    # Align HTF indicators to 4h timeframe
    high_vol_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_median_20[i]) or np.isnan(high_vol_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish Alligator alignment AND price above Donchian high AND HTF high volume regime
            if (bullish_alignment[i] and 
                close[i] > donchian_high[i] and 
                high_vol_regime_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish Alligator alignment AND price below Donchian low AND HTF high volume regime
            elif (bearish_alignment[i] and 
                  close[i] < donchian_low[i] and 
                  high_vol_regime_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on Alligator crossover (trend weakening)
            # Exit conditions: Jaw-Teeth crossover OR Teeth-Lips crossover
            exit_long = (position == 1 and 
                        ((jaw[i] >= teeth[i]) or (teeth[i] >= lips[i])))
            exit_short = (position == -1 and 
                         ((jaw[i] <= teeth[i]) or (teeth[i] <= lips[i])))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals