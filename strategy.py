#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and ATR-based stoploss
# - Long when price breaks above H3 (1d Camarilla resistance) AND 1d volume > 1.8x 20-period volume SMA
# - Short when price breaks below L3 (1d Camarilla support) AND 1d volume > 1.8x 20-period volume SMA
# - Exit: price retouches the pivot point (mean reversion within the day) OR ATR trailing stop
# - Uses 12h for price action and entry timing, 1d for Camarilla levels and volume confirmation
# - Target: 12-30 trades/year to minimize fee drag while capturing high-probability breakout moves
# - Camarilla pivots work well in ranging markets; volume confirmation reduces false breakouts

name = "12h_1d_camarilla_volspike_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels and volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: H4 = C + (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, H2 = C + (H-L)*1.1/6
    #          L3 = C - (H-L)*1.1/4, L2 = C - (H-L)*1.1/6, L1 = C - (H-L)*1.1/2
    #          Pivot = (H+L+C)/3
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN for first bar
    prev_high = np.where(np.isnan(prev_high), prev_close, prev_high)
    prev_low = np.where(np.isnan(prev_low), prev_close, prev_low)
    prev_close = np.where(np.isnan(prev_close), close[0] if len(close) > 0 else 0, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    h3 = pivot + (range_hl * 1.1 / 4)   # H3 resistance level
    l3 = pivot - (range_hl * 1.1 / 4)   # L3 support level
    pivot_level = pivot                 # Pivot point for exit
    
    # Align 1d levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ATR for stoploss (using 12h data)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 1.8x 20-period volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Long: price breaks above H3 resistance
            if close[i] > h3_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below L3 support
            elif close[i] < l3_aligned[i]:
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
            
            # Exit 1: price retouches pivot level (mean reversion)
            if position == 1 and close[i] <= pivot_aligned[i]:
                exit_signal = True
            elif position == -1 and close[i] >= pivot_aligned[i]:
                exit_signal = True
            
            # Exit 2: ATR trailing stop (2.5 * ATR)
            if not exit_signal and position != 0:
                # Track extreme prices for trailing stop (simplified: use entry-based stop)
                # In practice, we'd track highest high/lowest low since entry
                # For now, use a fixed stop from entry (approximation)
                if position == 1 and close[i] < close[i-1] - 2.5 * atr[i]:
                    exit_signal = True
                elif position == -1 and close[i] > close[i-1] + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals