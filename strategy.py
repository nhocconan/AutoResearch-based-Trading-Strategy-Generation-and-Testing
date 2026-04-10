#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H4 level AND 12h volume > 1.3x 20-period volume SMA AND chop < 61.8 (trending regime)
# - Short when price breaks below Camarilla L4 level AND 12h volume > 1.3x 20-period volume SMA AND chop < 61.8 (trending regime)
# - Exit: opposite Camarilla breakout or chop > 61.8 (range regime)
# - Uses 4h for price action and Camarilla levels, 12h for volume confirmation, 4h for chop regime
# - Camarilla pivots provide institutional support/resistance levels that work in both bull and bear markets
# - Volume confirmation ensures breakouts have conviction
# - Chop regime filter avoids false signals in ranging markets
# - Target: 20-50 trades/year to minimize fee drag while capturing meaningful moves

name = "4h_12h_camarilla_breakout_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute volume SMA for 12h data (20-period)
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    # Pre-compute 4h data for Camarilla and chop calculations
    # Calculate Camarilla pivot levels for 4h data (using prior period's high/low/close)
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    camarilla_h4 = close + 1.1 * (high - low) / 2.0
    camarilla_l4 = close - 1.1 * (high - low) / 2.0
    
    # Pre-compute Chopiness Index for 4h data (14-period)
    # Chop = 100 * log10(sum(ATR(1)/14) / log10(TrueRange(14))) 
    # Simplified: Chop = 100 * log10(ATR(14) / (TrueRange(14) * sqrt(14))) 
    # Even simpler approximation: Chop = 100 * log10(sum(abs(close - close.shift(1))/14) / log10(max(high, low) - min(high, low)) over 14)
    # We'll use a practical approximation: Chop = 100 * log10(ATR(14) / (TrueRange(14) * sqrt(14))) 
    # For simplicity, we'll use: Chop = 100 * log10(ATR(14) / (max(high, low) - min(high, low)) * sqrt(14))
    # Actually, let's use a more standard approach: Chop = 100 * log10(sum(ATR(1)/14) / log10(TrueRange(14)))
    # We'll implement a workable version:
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop Index: 100 * log10(sum(ATR(1)/14) / log10(TrueRange(14)))
    # We'll approximate: Chop = 100 * log10(atr / (pd.Series(tr).rolling(14).mean().values * np.sqrt(14)))
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr / (tr_ma * np.sqrt(14) + 1e-10))  # Add small epsilon to avoid division by zero
    chop = np.where(chop < 0, 100, chop)  # Ensure chop is between 0-100
    chop = np.where(chop > 100, 100, chop)
    
    for i in range(20, n):  # Start after 20-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20_12h_aligned[i]
        
        # Chop regime filter: chop < 61.8 indicates trending regime (good for breakouts)
        trending_regime = chop[i] < 61.8
        
        # Camarilla breakout signals
        breakout_long = close[i] > camarilla_h4[i-1]  # Break above prior period's H4
        breakout_short = close[i] < camarilla_l4[i-1]  # Break below prior period's L4
        
        # Exit conditions: opposite breakout or chop > 61.8 (range regime)
        exit_long = close[i] < camarilla_l4[i-1] or chop[i] > 61.8
        exit_short = close[i] > camarilla_h4[i-1] or chop[i] > 61.8
        
        # Trading logic
        if vol_confirm and trending_regime:
            # Long: Camarilla breakout above H4
            if breakout_long:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Camarilla breakout below L4
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
            # No volume confirmation or not trending: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals