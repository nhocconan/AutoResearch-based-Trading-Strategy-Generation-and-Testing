#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h volume spike + ATR filter
# - Primary: 4h price breaks above/below Camarilla H3/L3 levels from prior 1d session
# - HTF: 12h volume > 2.0x 20-period MA for confirmation (avoids low-volume breakouts)
# - Risk filter: ATR(14) < 0.03 * close (avoid extremely high volatility periods)
# - Long: Price breaks above H3 + volume confirmation + ATR filter
# - Short: Price breaks below L3 + volume confirmation + ATR filter
# - Exit: Price returns to prior day's close (mean reversion to equilibrium)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Camarilla adapts to volatility, volume filters false signals, ATR filter avoids chaotic markets
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_12h_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 25 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 1d data (prior session)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h ATR(14) for volatility filter
    atr = np.full(len(close_4h), np.nan)
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        if not np.isnan(volume_12h[i-19:i+1]).any():
            volume_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Calculate Camarilla levels from prior 1d session
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3 = np.full(len(close_4h), np.nan)
    camarilla_l3 = np.full(len(close_4h), np.nan)
    prior_close = np.full(len(close_4h), np.nan)
    
    for i in range(len(close_4h)):
        # Get prior 1d session data (use previous completed 1d bar)
        idx_1d = i // 96  # 96 = 4h bars per day (24h*60/15min)
        if idx_1d > 0 and idx_1d < len(high_1d):
            camarilla_h3[i] = close_1d[idx_1d-1] + 1.1 * (high_1d[idx_1d-1] - low_1d[idx_1d-1]) / 4
            camarilla_l3[i] = close_1d[idx_1d-1] - 1.1 * (high_1d[idx_1d-1] - low_1d[idx_1d-1]) / 4
            prior_close[i] = close_1d[idx_1d-1]
    
    # Align all HTF/LTF indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, prices, atr)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(atr_aligned[i]) or np.isnan(volume_ma_20_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(prior_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-period MA
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = volume_12h_aligned[i] > 2.0 * volume_ma_20_12h_aligned[i]
        
        # ATR filter: avoid extremely high volatility (ATR < 3% of price)
        atr_filter = atr_aligned[i] < 0.03 * close_4h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Camarilla H3 + volume confirmation + ATR filter
            if close_4h[i] > camarilla_h3_aligned[i] and volume_confirm and atr_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 + volume confirmation + ATR filter
            elif close_4h[i] < camarilla_l3_aligned[i] and volume_confirm and atr_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to prior day's close (mean reversion to equilibrium)
            if position == 1:  # Long position
                if close_4h[i] <= prior_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] >= prior_close_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals