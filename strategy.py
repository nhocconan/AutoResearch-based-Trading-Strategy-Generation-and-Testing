#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d ATR(14) > 20-period ATR SMA (high volatility regime) AND 12h volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) low AND 1d ATR(14) > 20-period ATR SMA AND 12h volume > 1.5x 20-period volume SMA
# - Exit: opposite Donchian breakout or volatility drops (ATR < ATR SMA) or volume < volume SMA
# - Uses 12h for price action and volume, 1d for volatility filter
# - ATR filter ensures we only trade during high volatility regimes, reducing whipsaws
# - Volume confirmation ensures breakouts have conviction
# - Donchian breakouts capture sustained moves in both bull and bear markets
# - Target: 12-30 trades/year to minimize fee drag while capturing meaningful moves

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for ATR calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate ATR for 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_sma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR SMA to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_sma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20)
    
    # Pre-compute Donchian channels for 12h data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR(14) > 20-period ATR SMA (high volatility regime)
        vol_filter = atr_14_aligned[i] > atr_sma_20_aligned[i]
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_high[i-1]  # Break above prior period's high
        breakout_short = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Exit conditions: opposite breakout or volatility drops or volume < average
        exit_long = (close[i] < donchian_low[i-1]) or (not vol_filter) or (volume[i] < volume_sma_20[i])
        exit_short = (close[i] > donchian_high[i-1]) or (not vol_filter) or (volume[i] < volume_sma_20[i])
        
        # Trading logic
        if vol_filter and vol_confirm:
            # Long: Donchian breakout above in high volatility regime
            if breakout_long:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below in high volatility regime
            elif breakout_short:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and exit_long:
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volatility filter or volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals