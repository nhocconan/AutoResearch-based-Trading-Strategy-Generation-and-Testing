#!/usr/bin/env python3
"""
1d_TRIX_VolumeSpike_RegimeFilter_v1
Hypothesis: Use TRIX (15-period) momentum with volume spike confirmation and choppiness regime filter on daily timeframe.
TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Volume spike (>1.5x 20-day average) confirms conviction.
Choppiness index (CHOP) > 61.8 = ranging market (mean reversion), CHOP < 38.2 = trending (follow TRIX).
In ranging markets: fade TRIX extremes (short when TRIX > 0.1, long when TRIX < -0.1).
In trending markets: follow TRIX (long when TRIX > 0, short when TRIX < 0).
Position size: 0.25 to balance risk and return.
Target: 15-25 trades/year to stay under 150 total trades hard max for 1d.
Works in bull (trending follow) and bear (mean reversion in ranges) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (optional, can be removed if not needed)
    # df_1w = get_htf_data(prices, '1w')
    
    # Calculate TRIX (15-period) - triple EMA of ROC
    # ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Volume spike: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index (CHOP) - 14-period
    # True Range = max(high-low, abs(high-close.prev), abs(low-close.prev))
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_14 * 14) / np.log10((highest_high_14 - lowest_low_14) + 1e-10)
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop_raw, 50.0)  # default to neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX (15*3=45), volume MA (20), CHOP (14) -> max 45
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime determination
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Neutral zone (38.2 <= CHOP <= 61.8) - use trending logic as default
        
        if position == 0:
            # Entry logic
            long_signal = False
            short_signal = False
            
            if is_ranging:
                # In ranging markets: mean reversion - fade TRIX extremes
                if trix[i] < -0.1 and volume_spike[i]:
                    long_signal = True
                elif trix[i] > 0.1 and volume_spike[i]:
                    short_signal = True
            else:
                # In trending or neutral markets: follow TRIX
                if trix[i] > 0 and volume_spike[i]:
                    long_signal = True
                elif trix[i] < 0 and volume_spike[i]:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: opposite TRIX signal with volume spike OR regime change to ranging with extreme TRIX
            if ((trix[i] < -0.1 and volume_spike[i]) or 
                (is_ranging and trix[i] > 0.1 and volume_spike[i])):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: opposite TRIX signal with volume spike OR regime change to ranging with extreme TRIX
            if ((trix[i] > 0.1 and volume_spike[i]) or 
                (is_ranging and trix[i] < -0.1 and volume_spike[i])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_TRIX_VolumeSpike_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0