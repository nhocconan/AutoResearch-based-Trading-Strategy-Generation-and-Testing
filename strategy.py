#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period volume SMA AND chop > 61.8 (range)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period volume SMA AND chop > 61.8 (range)
# - Exit: ATR trailing stop (2.5 * ATR from extreme) or opposite Camarilla level (H3/L3)
# - Uses 4h for price action and Camarilla levels, 1d for volume and chop regime
# - Camarilla pivots work well in ranging markets; volume spike confirms institutional interest
# - Choppiness filter ensures we only trade in ranging regimes where mean reversion works
# - ATR trailing stop manages risk while allowing trends to develop
# - Target: 20-30 trades/year to minimize fee drag while capturing high-probability mean reversion
# - Proven pattern: Camarilla + volume + chop filter worked well for ETHUSDT in DB (test Sharpe 1.47)

name = "4h_1d_camarilla_volchop_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(np.sum(atr_1d[-14:]) / np.log10(highest_high[-14:] - lowest_low[-14:])) if False else np.zeros_like(close_1d)  # placeholder
    # Proper CHOP calculation (vectorized)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_14 = highest_high - lowest_low
    chop_1d = 100 * np.log10(sum_tr_14 / np.log10(hh_ll_14))  # This is incorrect formula, fixing below
    # Correct Choppiness Index: CHOP = 100 * LOG10(SUM(ATR14) / LOG10(HH14 - LL14))
    # Actually: CHOP = 100 * LOG10(SUM(ATR,14) / LOG10(MAX(HIGH,14) - MIN(LOW,14)))
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / np.log10(max_high_14 - min_low_14))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute Camarilla pivot levels for 4h data (using previous day's OHLC)
    # For intraday, we use previous 1d bar's OHLC to calculate today's Camarilla levels
    # Since we're on 4h timeframe, we need to align the 1d OHLC to each 4h bar
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: H4 = close + range * 1.1/2, H3 = close + range * 1.1/4, etc.
    # Actually standard Camarilla: 
    # H4 = close + range * 1.1/2
    # H3 = close + range * 1.1/4
    # H2 = close + range * 1.1/6
    # H1 = close + range * 1.1/12
    # L1 = close - range * 1.1/12
    # L2 = close - range * 1.1/6
    # L3 = close - range * 1.1/4
    # L4 = close - range * 1.1/2
    range_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + range_1d * 1.1 / 4
    camarilla_l3_1d = close_1d - range_1d * 1.1 / 4
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
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
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_confirm = vol_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: Choppiness > 61.8 (ranging market)
        chop_confirm = chop_1d_aligned[i] > 61.8
        
        # Only trade when both volume and regime confirmations are present
        if vol_confirm and chop_confirm:
            # Update trailing stop extremes
            if position == 1:
                long_high[i] = max(long_high[i-1] if not np.isnan(long_high[i-1]) else close[i], close[i])
            elif position == -1:
                short_low[i] = min(short_low[i-1] if not np.isnan(short_low[i-1]) else close[i], close[i])
            else:
                long_high[i] = close[i]
                short_low[i] = close[i]
            
            # Long: price breaks above Camarilla H3
            if close[i] > camarilla_h3_1d_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Camarilla L3
            elif close[i] < camarilla_l3_1d_aligned[i]:
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
            
            # Exit 2: Opposite Camarilla level (reversal signal)
            if not exit_signal:
                if position == 1 and close[i] < camarilla_l3_1d_aligned[i]:
                    exit_signal = True
                elif position == -1 and close[i] > camarilla_h3_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals