#3.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Bollinger Bands for volatility and mean reversion, 
# combined with 1d RSI for momentum confirmation. Uses Bollinger Band width 
# to identify low volatility regimes (squeeze) and enters on mean reversion 
# from Bollinger Bands when RSI is oversold/overbought. 
# Designed to work in both bull and bear markets by focusing on mean reversion 
# during low volatility periods, which occur in all market conditions.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    close_1d = df_1d['close'].values
    
    basis = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    dev = bb_std * pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = basis + dev
    lower_band = basis - dev
    
    # Calculate RSI (14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (upper_band - lower_band) / basis
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    bb_width_ma_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ma)
    
    # Volume confirmation on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(bb_period, rsi_period, 50)  # Need BB, RSI, and BB width MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(basis_aligned[i]) or 
            np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or
            np.isnan(bb_width_ma_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Bollinger Band squeeze condition: low volatility regime
        volatility_squeeze = bb_width_aligned[i] < 0.8 * bb_width_ma_aligned[i]
        
        if position == 0:
            # Look for mean reversion entries during low volatility
            # Long: price touches lower BB AND RSI oversold (<30) AND volatility squeeze
            if (close[i] <= lower_band_aligned[i] and 
                rsi_aligned[i] < 30 and 
                volatility_squeeze and
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price touches upper BB AND RSI overbought (>70) AND volatility squeeze
            elif (close[i] >= upper_band_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  volatility_squeeze and
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to BB middle or RSI reaches neutral
            if (close[i] >= basis_aligned[i] or 
                rsi_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to BB middle or RSI reaches neutral
            if (close[i] <= basis_aligned[i] or 
                rsi_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dBollingerBands_RSI_Squeeze_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0