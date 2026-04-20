#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Bollinger Band mean reversion and volume confirmation.
# In sideways markets, price tends to revert to the mean after touching Bollinger Bands.
# Volume spike confirms the reversal signal. Works in both bull and bear markets by
# capturing mean reversion moves within larger trends.
name = "6h_1d_BollingerMeanReversion_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Bollinger Bands (20, 2) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower.values)
    
    # === 6h: Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20.values, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        vol_ratio_val = vol_ratio.values[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below lower BB with volume spike
            if (close_val <= lower and vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above upper BB with volume spike
            elif (close_val >= upper and vol_ratio_val > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle (SMA) or for risk management
            if close_val >= sma_20.iloc[-1] if not np.isnan(sma_20.iloc[-1]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle (SMA)
            if close_val <= sma_20.iloc[-1] if not np.isnan(sma_20.iloc[-1]) else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Fix: Calculate sma_20 properly for use in exit condition
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Bollinger Bands (20, 2) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower.values)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20.values)
    
    # === 6h: Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20.values, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        sma_val = sma_20_aligned[i]
        vol_ratio_val = vol_ratio.values[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(sma_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below lower BB with volume spike
            if (close_val <= lower and vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above upper BB with volume spike
            elif (close_val >= upper and vol_ratio_val > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle (SMA)
            if close_val >= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle (SMA)
            if close_val <= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Final version with proper variable scope
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Bollinger Bands (20, 2) ===
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower.values)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20.values)
    
    # === 6h: Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20.values, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        sma_val = sma_20_aligned[i]
        vol_ratio_val = vol_ratio.values[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(sma_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below lower BB with volume spike
            if (close_val <= lower and vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above upper BB with volume spike
            elif (close_val >= upper and vol_ratio_val > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle (SMA)
            if close_val >= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle (SMA)
            if close_val <= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_BollingerMeanReversion_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Bollinger Bands (20, 2) ===
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper.values)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower.values)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20.values)
    
    # === 6h: Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20.values, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        sma_val = sma_20_aligned[i]
        vol_ratio_val = vol_ratio.values[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(sma_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below lower BB with volume spike
            if (close_val <= lower and vol_ratio_val > 2.5):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above upper BB with volume spike
            elif (close_val >= upper and vol_ratio_val > 2.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to middle (SMA)
            if close_val >= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to middle (SMA)
            if close_val <= sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals