# [54090] Hypothesis: 1d timeframe strategy using 1-week Williams Alligator (3 SMAs: Jaw/Teeth/Lips) for trend direction and alignment, combined with 1d price crossing above/below the Teeth line for entry, with volume confirmation and ATR-based volatility filter. Designed to work in both bull (trend following) and bear (mean reversion during pullbacks) regimes by only taking trades when the Alligator is "awake" (diverged SMAs) and price is in alignment with the trend. Weekly timeframe ensures low-frequency, high-conviction signals to avoid overtrading.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator components (using 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: 3 SMAs shifted into the future
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    # We calculate on 1d close, then align and shift appropriately
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to lower timeframe (1d is our base, so no alignment needed for 1d->1d)
    # But we need to shift the SMAs to the right to avoid look-ahead
    jaw_1d_shifted = np.roll(jaw_1d, 8)
    teeth_1d_shifted = np.roll(teeth_1d, 5)
    lips_1d_shifted = np.roll(lips_1d, 3)
    # Set NaN for the shifted values that look ahead
    jaw_1d_shifted[:8] = np.nan
    teeth_1d_shifted[:5] = np.nan
    lips_1d_shifted[:3] = np.nan
    
    # Get 1w data for trend filter (EMA50) - using weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Alligator, EMA50, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_1d_shifted[i]) or 
            np.isnan(teeth_1d_shifted[i]) or 
            np.isnan(lips_1d_shifted[i]) or 
            np.isnan(ema50_1d[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator "sleeping" check: all three lines intertwined (market ranging)
        # We consider it "awake" if jaws < teeth < lips (for uptrend) OR jaws > teeth > lips (for downtrend)
        jaw = jaw_1d_shifted[i]
        teeth = teeth_1d_shifted[i]
        lips = lips_1d_shifted[i]
        
        # Avoid division by zero or near-zero in case of convergence
        if jaw == 0 and teeth == 0 and lips == 0:
            alligator_awake = False
        else:
            # Check if lines are properly ordered (not intertwined)
            alligator_awake = ((jaw < teeth and teeth < lips) or (jaw > teeth and teeth > lips))
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr14[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: Price above Teeth AND alligator awake (uptrend alignment) AND price above weekly EMA50
            if (close[i] > teeth_1d_shifted[i] and 
                alligator_awake and 
                jaw_1d_shifted[i] < teeth_1d_shifted[i] < lips_1d_shifted[i] and  # confirmed uptrend alignment
                close[i] > ema50_1d[i] and
                volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below Teeth AND alligator awake (downtrend alignment) AND price below weekly EMA50
            elif (close[i] < teeth_1d_shifted[i] and 
                  alligator_awake and 
                  jaw_1d_shifted[i] > teeth_1d_shifted[i] > lips_1d_shifted[i] and  # confirmed downtrend alignment
                  close[i] < ema50_1d[i] and
                  volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Teeth OR alligator starts sleeping (lines intertwine)
            if (close[i] < teeth_1d_shifted[i]) or not alligator_awake:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Teeth OR alligator starts sleeping
            if (close[i] > teeth_1d_shifted[i]) or not alligator_awake:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_EMA50_Trend"
timeframe = "1d"
leverage = 1.0