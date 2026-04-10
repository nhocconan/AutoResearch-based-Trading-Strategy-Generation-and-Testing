#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 1d ATR(14) > 1.5x 20-period ATR SMA AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Donchian(20) lower band AND 1d ATR(14) > 1.5x 20-period ATR SMA AND volume > 1.5x 20-period volume SMA
# - Exit: ATR trailing stop (2.5 * ATR from extreme) or opposite Donchian band break
# - Uses 4h for price action and Donchian channels, 1d for volatility and volume filters
# - Volatility filter ensures we trade during expanded volatility regimes (works in both bull and bear markets)
# - Volume confirmation adds conviction to breakouts
# - ATR trailing stop manages risk while letting winners run
# - Target: 20-35 trades/year to minimize fee drag while capturing high-probability trends

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for volatility and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d ATR for volatility filter
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR SMA for volatility filter reference
    atr_sma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_sma_20_1d)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Donchian channels for 4h data (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for trailing stop (using 4h data)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track extreme prices for trailing stop
    long_high = np.full(n, np.nan)
    short_low = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_sma_20_1d_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d ATR and volume (aligned)
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volatility filter: 1d ATR > 1.5x 20-period ATR SMA
        vol_filter = atr_1d_aligned[i] > 1.5 * atr_sma_20_1d_aligned[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Only trade when both volatility and volume confirmation are present
        if vol_filter and vol_confirm:
            # Update trailing stop extremes
            if position == 1:
                long_high[i] = max(long_high[i-1] if not np.isnan(long_high[i-1]) else close[i], close[i])
            elif position == -1:
                short_low[i] = min(short_low[i-1] if not np.isnan(short_low[i-1]) else close[i], close[i])
            else:
                long_high[i] = close[i]
                short_low[i] = close[i]
            
            # Long: price breaks above Donchian upper band
            if close[i] > donchian_upper[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Donchian lower band
            elif close[i] < donchian_lower[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions
            exit_signal = False
            
            # Exit 1: ATR trailing stop (2.5 * ATR from extreme)
            if position == 1 and not np.isnan(long_high[i]):
                if close[i] < long_high[i] - 2.5 * atr[i]:
                    exit_signal = True
            elif position == -1 and not np.isnan(short_low[i]):
                if close[i] > short_low[i] + 2.5 * atr[i]:
                    exit_signal = True
            
            # Exit 2: Opposite Donchian band break (reversal signal)
            if not exit_signal:
                if position == 1 and close[i] < donchian_lower[i]:
                    exit_signal = True
                elif position == -1 and close[i] > donchian_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No volatility or volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals