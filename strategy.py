#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 4h Donchian breakout
# - Choppiness Index (CHOP) > 61.8 = ranging market (mean reversion)
# - CHOP < 38.2 = trending market (trend following)
# - In ranging: long when price touches lower Bollinger Band (20,2), short when touches upper band
# - In trending: long when price breaks above Donchian upper (20), short when breaks below lower
# - Volume confirmation: volume > 1.3x 20-period average
# - ATR-based stop: exit when price moves 2.5x ATR against position
# - Uses only 4h timeframe for simplicity and robustness
# - Target: 30-50 trades per year per symbol (120-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss and choppiness
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR over n) / (max(high,n) - min(low,n))) / log10(n)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    # Avoid division by zero
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    
    # Donchian Channel (20) for trend following in trending markets
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(chop[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Determine market regime
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            if is_ranging:
                # Mean reversion in ranging market
                if price <= lower_bb[i] and vol > 1.3 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price >= upper_bb[i] and vol > 1.3 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif is_trending:
                # Trend following in trending market
                if price > donch_high[i] and vol > 1.3 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif price < donch_low[i] and vol > 1.3 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            if is_ranging and price >= ma20[i]:  # Return to mean in ranging
                exit_signal = True
            elif is_trending and price < donch_low[i]:  # Breakdown in trending
                exit_signal = True
            elif price <= entry_price - 2.5 * atr[i]:  # ATR stop
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            if is_ranging and price <= ma20[i]:  # Return to mean in ranging
                exit_signal = True
            elif is_trending and price > donch_high[i]:  # Breakout in trending
                exit_signal = True
            elif price >= entry_price + 2.5 * atr[i]:  # ATR stop
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChopRegime_DonchianBB_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0