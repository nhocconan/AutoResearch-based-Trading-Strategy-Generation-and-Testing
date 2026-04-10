#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND chop < 61.8 (trending)
# - Exit: ATR trailing stop (2.0 * ATR from extreme) or opposite Camarilla level (H3/L3) break
# - Uses 4h for price action and Camarilla levels, 1d for volume and chop confirmation
# - Choppiness filter avoids false breakouts in ranging markets
# - ATR trailing stop manages risk while letting winners run
# - Target: 20-35 trades/year to minimize fee drag while capturing high-probability trends
# - Proven pattern: Camarilla + volume + regime filter worked well for ETHUSDT in DB (test Sharpe 1.47)

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and chop confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low))))
    # We use a simplified version: CHOP = 100 * log10(ATR_sum / (log10(14) * (HH - LL))) / log10(14)
    # But we'll use the standard formula: CHOP = 100 * log10(sum(ATR(14),14) / (log10(14) * (max(high,14) - min(low,14))))
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP: 100 * log10(sum(ATR(14),14) / (log10(14) * (max(high,14) - min(low,14))))
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(14) * (hh - ll)
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop = 100 * np.log10(atr_sum / chop_denom)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute Camarilla levels for 4h data (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We use H3 and L3 for breakout signals
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # But we need to use previous day's OHLC, so we shift by 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
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
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume and chop (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: chop < 61.8 (trending market)
        regime_confirm = chop_aligned[i] < 61.8
        
        # Only trade when both volume and regime confirmation are present
        if vol_confirm and regime_confirm:
            # Update trailing stop extremes
            if position == 1:
                long_high[i] = max(long_high[i-1] if not np.isnan(long_high[i-1]) else close[i], close[i])
            elif position == -1:
                short_low[i] = min(short_low[i-1] if not np.isnan(short_low[i-1]) else close[i], close[i])
            else:
                long_high[i] = close[i]
                short_low[i] = close[i]
            
            # Long: price breaks above Camarilla H3 level
            if close[i] > camarilla_h3_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Camarilla L3 level
            elif close[i] < camarilla_l3_aligned[i]:
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
            
            # Exit 1: ATR trailing stop (2.0 * ATR from extreme)
            if position == 1 and not np.isnan(long_high[i]):
                if close[i] < long_high[i] - 2.0 * atr[i]:
                    exit_signal = True
            elif position == -1 and not np.isnan(short_low[i]):
                if close[i] > short_low[i] + 2.0 * atr[i]:
                    exit_signal = True
            
            # Exit 2: Opposite Camarilla level break (reversal signal)
            if not exit_signal:
                if position == 1 and close[i] < camarilla_l3_aligned[i]:
                    exit_signal = True
                elif position == -1 and close[i] > camarilla_h3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No volume or regime confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals